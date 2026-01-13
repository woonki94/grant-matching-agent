from langchain_core.prompts import ChatPromptTemplate

FACULTY_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from faculty context for later structuring.\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- Return a single JSON object with key `candidates` = list[str].\n"
     "- Include TWO kinds of candidates:\n"
     "  (A) AREA terms: 1–3 words, broad fields/areas that define the work (e.g., 'heat transfer', 'robotics').\n"
     "  (B) SPECIALIZATION statements: 8–25 words, detailed phrases capturing methods/problems/systems (can be sentence-like).\n"
     "- Prefer technical phrases over generic admin words.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- Keep to ~30–80 candidates total."),
    ("human", "Context (JSON):\n{context_json}")
])

OPP_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from a funding opportunity context for later structuring.\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- Return a single JSON object with key `candidates` = list[str].\n"
     "- Include TWO kinds of candidates:\n"
     "  (A) AREA terms: 1–3 words, broad research areas the opportunity targets.\n"
     "  (B) SPECIALIZATION statements: 8–25 words, detailed phrases describing goals, methods, constraints, or targeted problems.\n"
     "- Prefer domain terms and technical content; avoid application paperwork language.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- Keep to ~30–80 candidates total."),
    ("human", "Context (JSON):\n{context_json}")
])

FACULTY_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate structured keywords from faculty context.\n"
     "Output must match the schema:\n"
     "{{\"research\": {{\"domain\": [], \"specialization\": []}}, "
     "\"application\": {{\"domain\": [], \"specialization\": []}}}}\n"
     "Definitions:\n"
     "- domain (AREA): 1–3 words that define the work area/field. Keep short and general.\n"
     "- specialization (SPEC): 8–25 words, detailed keyword statements describing methods/problems/systems; may be sentence-like.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context + candidate phrases.\n"
     "- Do NOT invent new topics.\n"
     "- Lowercase except proper nouns.\n"
     "- Deduplicate.\n"
     "- research.domain: 3–8 items\n"
     "- research.specialization: 3–10 items\n"
     "- application.domain: 2–6 items (sectors/contexts)\n"
     "- application.specialization: 2–8 items (detailed applied statements)\n"
     "- If unsure, leave lists shorter rather than guessing.\n"
     "- Avoid admin terms: 'grant', 'university', 'department', 'proposal'."),
    ("human",
     "Context (JSON):\n{context_json}\n\n"
     "Candidate phrases:\n{candidates}")
])

OPP_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate structured keywords from a funding opportunity context.\n"
     "Output must match the schema:\n"
      "{{\"research\": {{\"domain\": [], \"specialization\": []}}, "
     "\"application\": {{\"domain\": [], \"specialization\": []}}}}\n"
     "Definitions:\n"
     "- domain (AREA): 1–3 words that define the research areas targeted.\n"
     "- specialization (SPEC): 8–25 words, detailed keyword statements capturing scope, methods, constraints, or targeted problems.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context + candidate phrases.\n"
     "- Do NOT invent requirements.\n"
     "- Lowercase except proper nouns.\n"
     "- Deduplicate.\n"
     "- research.domain: 3–10 items\n"
     "- research.specialization: 3–12 items\n"
     "- application.domain: 2–8 items (sectors/beneficiaries)\n"
     "- application.specialization: 2–10 items (detailed applied statements)\n"
     "- Avoid generic funding language ('submit', 'eligibility', 'deadline') unless it encodes a real technical constraint."),
    ("human",
     "Context (JSON):\n{context_json}\n\n"
     "Candidate phrases:\n{candidates}")
])