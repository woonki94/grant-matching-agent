from langchain_core.prompts import ChatPromptTemplate



FACULTY_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from faculty context for later structuring.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- Return a single JSON object with key `candidates` = list[str].\n"
     "- Include TWO kinds of candidates:\n"
     "  (A) AREA terms: 1–3 words, broad research fields or domains.\n"
     "  (B) SPECIALIZATION statements: 8–25 words describing expertise, methods, systems, or problems the faculty works on.\n"
     "- SPECIALIZATION phrases must describe capabilities the faculty HOLDS.\n"
     "- Write phrases neutrally (no 'seeks', 'needs', 'requires').\n"
     "- Prefer technical and research content; avoid administrative language.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- Target ~30–80 candidates total."
    ),
    ("human", "Context (JSON):\n{context_json}")
])

OPP_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from a funding opportunity context for later structuring.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- Return a single JSON object with key `candidates` = list[str].\n"
     "- Include TWO kinds of candidates:\n"
     "  (A) AREA terms: 1–3 words, broad research or technical areas targeted.\n"
     "  (B) SPECIALIZATION statements: 8–25 words describing capabilities, methods, or expertise the project expects investigators to have.\n"
     "- SPECIALIZATION phrases must describe what the grant NEEDS faculty to have.\n"
     "- Write phrases factually (do not include submission or administrative language).\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- Target ~30–80 candidates total."
    ),
    ("human", "Context (JSON):\n{context_json}")
])

FACULTY_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate structured keywords from faculty context.\n\n"
     "Output must match this JSON schema:\n"
     "{{\n"
     "  \"research\": {{\"domain\": [], \"specialization\": []}},\n"
     "  \"application\": {{\"domain\": [], \"specialization\": []}}\n"
     "}}\n\n"
     "Definitions:\n"
     "- domain: 1–3 words, broad fields defining the faculty’s work.\n"
     "- specialization: 8–25 words describing expertise, methods, or systems the faculty HAS.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context and candidate phrases.\n"
     "- Do NOT invent new topics.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- research.domain: 4–10 items\n"
     "- research.specialization: 5–15 items\n"
     "- application.domain: 4–10 items\n"
     "- application.specialization: 5–15 items\n"
     "- Avoid administrative or institutional language."
    ),
    ("human",
     "Context (JSON):\n{context_json}\n\n"
     "Candidate phrases:\n{candidates}"
    )
])


OPP_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate structured keywords from a funding opportunity context.\n\n"
     "Output must match this JSON schema:\n"
     "{{\n"
     "  \"research\": {{\"domain\": [], \"specialization\": []}},\n"
     "  \"application\": {{\"domain\": [], \"specialization\": []}}\n"
     "}}\n\n"
     "Definitions:\n"
     "- domain: 1–3 words, broad research or technical areas.\n"
     "- specialization: 8–25 words describing expertise, methods, or capabilities the project EXPECTS investigators to have.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context and candidate phrases.\n"
     "- Do NOT invent requirements not present in the text.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- research.domain: 4–10 items\n"
     "- research.specialization: 5–15 items\n"
     "- application.domain: 4–10 items\n"
     "- application.specialization: 5–15 items\n"
     "- Avoid generic submission or eligibility language."
    ),
    ("human",
     "Context (JSON):\n{context_json}\n\n"
     "Candidate phrases:\n{candidates}"
    )
])

FACULTY_SPECIALIZATION_WEIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign expertise weights to specialization phrases for a faculty member.\n\n"
     "You will be given:\n"
     "- faculty context (JSON)\n"
     "- specialization lists for research and application\n\n"
     "Important clarification:\n"
     "- The given specialization phrases represent areas that the faculty member is professionally experienced in.\n\n"
     "Task:\n"
     "- For EACH specialization phrase, output an object with:\n"
     "  - t: the exact original specialization text (unchanged)\n"
     "  - w: a number in [0,1] representing the faculty's level of expertise/proficiency in that specialization\n\n"
     "Weight guidance:\n"
     "- 0.85–1.00: primary expertise, strong evidence in context\n"
     "- 0.60–0.84: strong working expertise\n"
     "- 0.35–0.59: some experience / occasional involvement\n"
     "- 0.10–0.34: weak/unclear evidence (use if mentioned but not supported)\n\n"
     "Rules:\n"
     "- Do NOT invent new specialization phrases.\n"
     "- Keep text exactly the same as input for t.\n"
     "- If evidence is unclear, be conservative (lower w).\n"
     "- Return ONLY JSON in the specified output format."
    ),
    ("human",
     "FACULTY CONTEXT (JSON):\n{context_json}\n\n"
     "SPECIALIZATIONS INPUT (JSON):\n{spec_json}\n\n"
     "Return weighted specializations JSON."
    )
])

OPP_SPECIALIZATION_WEIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign importance weights to specialization requirements for a funding opportunity.\n\n"
     "You will be given:\n"
     "- opportunity context (JSON)\n"
     "- specialization lists for research and application\n\n"
     "Important clarification:\n"
     "- The given specialization phrases represent capabilities that the opportunity REQUIRES faculty investigators to have.\n\n"
     "Task:\n"
     "- For EACH specialization phrase, output an object with:\n"
     "  - t: the exact original specialization text (unchanged)\n"
     "  - w: a number in [0,1] representing how critical this requirement is to the project’s success\n\n"
     "Weight guidance:\n"
     "- 0.85–1.00: core / essential requirement (project cannot succeed without it)\n"
     "- 0.60–0.84: important requirement (strongly preferred)\n"
     "- 0.35–0.59: supporting requirement\n"
     "- 0.10–0.34: minor or optional requirement\n\n"
     "Rules:\n"
     "- Do NOT invent new specialization phrases.\n"
     "- Keep the text exactly the same as input for t.\n"
     "- Base weights ONLY on evidence in the provided opportunity context.\n"
     "- If importance is unclear, be conservative (assign a lower weight).\n"
     "- Do NOT include administrative, eligibility, or submission criteria.\n"
     "- Return ONLY JSON in the specified output format."
    ),
    ("human",
     "OPPORTUNITY CONTEXT (JSON):\n{context_json}\n\n"
     "SPECIALIZATIONS INPUT (JSON):\n{spec_json}\n\n"
     "Return weighted specializations JSON."
    )
])

