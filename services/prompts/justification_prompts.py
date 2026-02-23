from langchain_core.prompts import ChatPromptTemplate

FACULTY_RECS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write faculty-facing recommendations for funding opportunities.\n"
     "Return ONLY JSON matching this schema:\n"
     "{{ \"faculty_name\": string, \"recommendations\": FacultyOpportunityRec[] }}.\n"
     "\n"
     "INPUT CONTEXT LIMITATION:\n"
     "- You only get faculty metadata+keywords and opportunity metadata+keywords.\n"
     "- Infer fit from keyword overlap and score fields; do not assume missing details.\n"
     "\n"
     "ABOUT SCORES (mention naturally in content):\n"
     "- domain_score: cosine similarity of domain embeddings (fast topical overlap filter).\n"
     "- llm_score: an LLM judgment score based on detailed fit between faculty context and opportunity context.\n"
     "- llm_score matters more than domain_score when interpreting fit.\n"
     "\n"
     "QUALITATIVE LABELS (based on llm_score; use ONE label per opportunity in your bullets):\n"
     "- llm_score < 0.30: 'mismatch' (weak fit)\n"
     "- 0.30–0.49: 'bad' (mostly mismatch)\n"
     "- 0.50–0.69: 'good' (reasonable fit)\n"
     "- 0.70–0.84: 'great' (strong fit)\n"
     "- >= 0.85: 'fantastic' (excellent fit)\n"
     "\n"
     "REQUIRED CONTENT PER OPPORTUNITY:\n"
     "- Include domain_score and llm_score in the output fields.\n"
     "- why_good_match MUST include:\n"
     "  (1) one bullet with qualitative label and keyword-based reason,\n"
     "  (2) 1–2 bullets on concrete topical alignment,\n"
     "  (3) one bullet on risk/gap and fallback strategy.\n"
     "\n"
     "STYLE + CONSTRAINTS:\n"
     "- Use ONLY the provided contexts.\n"
     "- If TOP OPPORTUNITIES is non-empty, recommendations MUST contain at least one item.\n"
     "- why_good_match: 3–5 bullets, each 8–18 words.\n"
     "- suggested_pitch: 1–2 sentences, <= 220 characters total.\n"
     "- Do not mention embeddings, cosine, vector databases, or OpenAI/OpenRouter.\n"
     "- You MAY mention that 'domain score is a topical overlap estimate' and 'LLM score is a deeper fit judgment'.\n"
    ),
    ("human",
     "FACULTY CONTEXT (JSON):\n{faculty_json}\n\n"
     "TOP OPPORTUNITIES (JSON list):\n{opps_json}\n")
])
