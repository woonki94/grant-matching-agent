from langchain_core.prompts import ChatPromptTemplate

FACULTY_RECS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write faculty-facing recommendations for funding opportunities.\n"
     "Return ONLY JSON matching the schema (a list of FacultyOpportunityRec objects).\n"
     "\n"
     "ABOUT SCORES (you must explain these in the content):\n"
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
     "  (1) one bullet that states the qualitative label and why (tie to context),\n"
     "  (2) 2–4 bullets on concrete alignment (topics/methods/problems),\n"
     "  (3) one bullet on risks/gaps (even if good/fantastic) + a fallback strategy,\n"
     "  (4) if label is 'sucks' or 'bad', include one bullet describing potential pivot angle.\n"
     "\n"
     "STYLE + CONSTRAINTS:\n"
     "- Use ONLY the provided contexts.\n"
     "- why_good_match: 5–8 bullets, each 12–22 words (keep them skimmable, not long).\n"
     "- suggested_pitch: 2–3 sentences, <= 240 characters total.\n"
     "- Do not mention embeddings, cosine, vector databases, or OpenAI/OpenRouter.\n"
     "- You MAY mention that 'domain score is a topical overlap estimate' and 'LLM score is a deeper fit judgment'.\n"
    ),
    ("human",
     "FACULTY CONTEXT (JSON):\n{faculty_json}\n\n"
     "TOP OPPORTUNITIES (JSON list):\n{opps_json}\n")
])