# Independent Agentic Architecture (Faculty -> Grant One-to-One)

This module is fully independent from `services/agent_v2`.

Path:
- `tmp/agentic_arch/`

## Agent flow
1. `FacultyProfessionAgent`
- Parallel tools: basic info, keywords, additional text, publications
- Output: `profession_focus` for the faculty

2. `GrantAgent`
- Find candidate grants for profession focus
- For each candidate grant, parallel calls:
  - grant metadata tool (`agency_name`, `close_date`, `grant_name`)
  - grant requirement agent

3. `GrantRequirementAgent`
- Parallel requirement sub-calls:
  - domains
  - specializations
  - eligibility
  - deliverables

4. `OneToOneProfessionMatcher`
- One-to-one ranking between faculty profession focus and grant requirements

## Files
- `models.py`: shared dataclasses
- `tools.py`: tool protocols + in-memory implementations
- `neo4j_tools.py`: real Neo4j-backed tool adapters for faculty/grant graphs
- `faculty_agent.py`: faculty profession inference agent
- `grant_agent.py`: grant retrieval + requirement parallel agent
- `matcher.py`: one-to-one scorer
- `orchestrator.py`: end-to-end orchestration
- `run_demo.py`: executable demo
- `run_neo4j_match.py`: executable Neo4j-backed one-to-one run

## Demo
```bash
python tmp/agentic_arch/run_demo.py
```

## Neo4j run
```bash
python tmp/agentic_arch/run_neo4j_match.py \
  --email "alan.fern@oregonstate.edu" \
  --candidate-grant-k 30 \
  --top-k 5
```
