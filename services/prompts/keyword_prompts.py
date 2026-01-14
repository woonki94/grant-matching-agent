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