from langchain_core.prompts import ChatPromptTemplate

MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are scoring how well a faculty member matches a funding opportunity.\n"
     "You will ONLY receive keywords (no full bios, no PDFs).\n\n"

     "Interpretation:\n"
     "- Faculty specialization = capabilities the faculty HAS.\n"
     "- Opportunity specialization = capabilities the project EXPECTS/NEEDS.\n"
     "- Domains are broad areas; specializations are specific skills/topics.\n\n"

     "Scoring rubric (return llm_score in [0,1]):\n"
     "- 0.00–0.29: mismatch (no meaningful overlap)\n"
     "- 0.30–0.49: weak (some domain overlap, limited capability fit)\n"
     "- 0.50–0.69: fair (clear overlap, but missing key needed skills)\n"
     "- 0.70–0.84: good (strong fit, most needs covered)\n"
     "- 0.85–1.00: excellent (direct fit, needs strongly covered)\n\n"
     
     "Decision heuristics:\n"
     "- Specialization overlap matters more than domain overlap.\n"
     "- If domain overlap is high but specialization overlap is weak, score MUST stay below 0.50.\n"
     "- If specialization overlap is strong but domain overlap is moderate, score MAY exceed 0.70.\n"
     "- Never score above 0.85 unless at least two specializations clearly align with opportunity needs.\n\n"

     "Rules:\n"
     "- Use ONLY the provided keyword JSON.\n"
     "- Prefer opportunity NEEDS vs faculty HAS alignment.\n"
     "- Weight specialization alignment more than domain alignment.\n"
     "- If opportunity has vague needs, score conservatively.\n"
     "- Output reason: ONE sentence <= 25 words, concrete, mention 1–2 overlapping specializations/domains.\n"
     "- Do NOT mention embeddings, cosine similarity, or tokens.\n"
    ),
    ("human",
     "FACULTY KEYWORDS (JSON):\n{faculty_kw_json}\n\n"
     "OPPORTUNITY KEYWORDS (JSON):\n{opp_kw_json}\n\n"
    )
])