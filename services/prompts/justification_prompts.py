from langchain_core.prompts import ChatPromptTemplate

FACULTY_RECS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You explain why a faculty member fits or does not fit a grant opportunity.\n"
        "Return ONLY valid JSON matching this exact schema:\n"
        '{{\n'
        '  "summary": string,\n'
        '  "alignment_points": [string, ...],\n'
        '  "risk_gaps": [string, ...]\n'
        '}}\n\n'
        "Rules:\n"
        "- summary: 1-2 sentences capturing the core match reason. Wrap the 2-3 most important terms in **double asterisks** (e.g. **federated learning**).\n"
        "- alignment_points: 2-4 specific strengths as short bullet strings. Each must cite a concrete piece of evidence (a publication title, research area, or funded project). Wrap key terms in **double asterisks**. Keep each bullet under 25 words.\n"
        "- risk_gaps: 1-3 specific gaps or weaknesses as short bullet strings. Be precise, not generic. Wrap key terms in **double asterisks**. Keep each bullet under 25 words.\n"
        "- Use ONLY the provided context. Do NOT invent evidence.\n"
        "- Do NOT include numeric scores, percentages, or the words 'requirement' or 'alignment'.\n"
        "- Do NOT output anything outside the JSON.\n"
    ),
    (
        "human",
        "SCORE HINT:\n{score_context}\n\n"
        "MATCH CONTEXT:\n{context_text}\n",
    ),
])


GRANT_EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You write a grant explanation from grant context JSON.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{{ \"grant_explanation\": string }}\n"
        "\n"
        "Rules:\n"
        "- Use only the provided context.\n"
        "- Explain what the grant is about, what it emphasizes, and what capabilities it expects.\n"
        "- Keep it clear and concrete.\n"
        "- Do not output anything outside JSON.\n"
    ),
    (
        "human",
        "GRANT CONTEXT (JSON):\n{grant_json}\n",
    ),
])


FACULTY_TOP_GRANT_RERANK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rerank top grant candidates for ONE faculty.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{{ \"ranked_opportunity_ids\": [string, ...], \"reranked_grants\": [{{\"opportunity_id\": string, \"llm_score\": float}}] }}\n"
        "\n"
        "Rules:\n"
        "- Use only the provided faculty and grant keyword inventories.\n"
        "- Rerank by domain + specialization fit quality.\n"
        "- Recompute llm_score between 0.0 and 1.0 for each grant.\n"
        "- ranked_opportunity_ids and reranked_grants must contain the same grant IDs exactly once.\n"
        "- Do not output text outside JSON.\n"
    ),
    (
        "human",
        "FACULTY (JSON):\n{faculty_json}\n\n"
        "GRANTS (JSON list):\n{grants_json}\n",
    ),
])
