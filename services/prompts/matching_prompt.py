from langchain_core.prompts import ChatPromptTemplate

MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are scoring how well a faculty member matches a funding opportunity.\n"
     "You will ONLY receive faculty keywords and an indexed list of opportunity requirements.\n\n"

     "Interpretation:\n"
     "- Faculty specialization = capabilities the faculty HAS.\n"
     "- Opportunity requirements = capabilities the project EXPECTS/NEEDS.\n"
     "- Requirements are grouped into two sections: application and research.\n\n"

     "Scoring rubric (return llm_score in [0,1]):\n"
     "- 0.00–0.29: mismatch (no meaningful overlap)\n"
     "- 0.30–0.49: weak (some overlap, limited capability fit)\n"
     "- 0.50–0.69: fair (clear overlap, but missing key needed skills)\n"
     "- 0.70–0.84: good (strong fit, most needs covered)\n"
     "- 0.85–1.00: excellent (direct fit, needs strongly covered)\n\n"

     "Decision heuristics:\n"
     "- Coverage of APPLICATION requirements matters more than RESEARCH requirements.\n"
     "- If few application requirements are covered, llm_score must stay below 0.50.\n"
     "- Never score above 0.85 unless at least two application requirements are clearly covered.\n\n"

     "Rules:\n"
     "- Use ONLY the provided faculty keywords and REQUIREMENTS_INDEXED.\n"
     "- reason: EXACTLY one sentence, <= 10 words, concrete, mention 1–2 overlapping capabilities.\n"
     "- Do NOT mention embeddings, cosine similarity, tokens, or prompting.\n\n"

     "Coverage output requirements (MUST follow exactly):\n"
     "- REQUIREMENTS_INDEXED is a JSON object with keys: 'application' and 'research'.\n"
     "- Each section maps string indices to requirement text, e.g. {{\"0\": \"...\", \"1\": \"...\"}}.\n"
     "- You MUST output two lists: 'covered' and 'missing'. Each element must be:\n"
     "  {{\"section\": \"application\"|\"research\", \"idx\": <int>}}.\n"
     "- idx is 0-based and must exist in the corresponding section.\n"
     "- Do NOT output requirement text in covered/missing — only section + idx.\n"
     "- No duplicates. Do not put the same (section, idx) in both lists.\n"
     "- Be conservative: if unsure, put it in missing.\n\n"
     "Completeness constraint (IMPORTANT):\n"
     "- For EACH section ('application' and 'research'), you must classify EVERY index that appears\n"
     "  in REQUIREMENTS_INDEXED[section] into exactly one of covered or missing.\n"
     "  (i.e., covered ∪ missing = all indices, and covered ∩ missing = ∅, per section).\n"

     "Final self-check:\n"
     "- Validate every (section, idx) exists in REQUIREMENTS_INDEXED before responding."
    ),
    ("human",
     "FACULTY KEYWORDS (JSON):\n{faculty_kw_json}\n\n"
     "REQUIREMENTS_INDEXED (JSON):\n{requirements_indexed}\n\n"
     "Return the structured fields only."
    )
])