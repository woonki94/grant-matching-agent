from langchain_core.prompts import ChatPromptTemplate

FACULTY_RECS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You write a single narrative explanation of why a faculty member matches a grant.\n"
        "Use only the provided context.\n"
        "Score note: llm_score is first produced from cross-encoder specialization matching, then reranked by an LLM.\n"
        "Treat llm_score as a lightweight confidence hint only.\n"
        "Write naturally in paragraph form, with concrete evidence from publications/profile chunks when available.\n"
        "Focus on the match reasoning only.\n"
        "Strict prohibition: do NOT include requirement labels/text, and do NOT include any numeric scores/weights/percentages.\n"
        "Return plain text explanation only.\n"
    ),
    (
        "human",
        "CURRENT SCORE HINT:\n{score_context}\n\n"
        "JUSTIFICATION CONTEXT:\n{context_text}\n",
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
