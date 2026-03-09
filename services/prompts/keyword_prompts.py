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

Faculty_CHUNK_RESEARCH_KEYWORD_LINK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract RESEARCH keywords from faculty chunk rows and attach specialization evidence ids.\n\n"
     "Input:\n"
     "- JSON list of chunk rows: [{{chunk_id, text}}, ...]\n\n"
     "Output JSON schema:\n"
     "{{\n"
     "  \"research\": {{\n"
     "    \"domain\": [\"...\"],\n"
     "    \"specialization\": [{{\"t\": \"...\", \"e\": {{\"chunk_id\": 0.0}}}}]\n"
     "  }},\n"
     "  \"application\": {{\n"
     "    \"domain\": [],\n"
     "    \"specialization\": []\n"
     "  }}\n"
     "}}\n\n"
     "IMPORTANT OUTPUT SHAPE:\n"
     "- specialization MUST be an array of objects, not strings.\n"
     "- Each specialization item MUST be exactly {{\"t\": string, \"e\": {{\"chunk_id\": confidence}}}}.\n"
     "- INVALID (do not do): [\"keyword text\"]\n"
     "- VALID: [{{\"t\": \"sim-to-real reinforcement learning for biped locomotion\", \"e\": {{\"abc|1|2\": 0.91}}}}]\n\n"
     "Rules:\n"
     "- Use ONLY provided chunk text.\n"
     "- domain: 2-4 words.\n"
     "- specialization: 8-20 words, expertise the faculty HAS.\n"
     "- Focus ONLY on research intent: methods/theory/algorithms/modeling/scientific inquiry.\n"
     "- research.domain: 4-8 items.\n"
     "- research.specialization: 5-9 items.\n"
     "- Write specialization as concise technical noun phrases (not sentences).\n"
     "- One specialization per distinct concept; do NOT emit paraphrase variants.\n"
     "- application must remain empty.\n"
     "- Lowercase unless proper nouns.\n"
     "- Do not invent chunk ids.\n"
     "- For specialization, e must be a non-empty object map chunk_id -> confidence in [0,1].\n"
     "- NEVER return specialization as list[str].\n"
     "- If evidence is missing for a candidate, omit it rather than returning invalid shape.\n"
     "- Domains do not require evidence ids.\n"
     "- Do not output empty keywords.\n"
     "- Keep output compact and deduplicated.\n"
     "- Confidence must be discriminative; do not assign one default value to all chunk ids."
    ),
    ("human", "Chunks JSON:\n{chunks_json}")
])

Faculty_CHUNK_APPLICATION_KEYWORD_LINK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract APPLICATION keywords from faculty chunk rows and attach specialization evidence ids.\n\n"
     "Input:\n"
     "- JSON list of chunk rows: [{{chunk_id, text}}, ...]\n\n"
     "Output JSON schema:\n"
     "{{\n"
     "  \"research\": {{\"domain\": [], \"specialization\": []}},\n"
     "  \"application\": {{\n"
     "    \"domain\": [\"...\"],\n"
     "    \"specialization\": [{{\"t\": \"...\", \"e\": {{\"chunk_id\": 0.0}}}}]\n"
     "  }}\n"
     "}}\n\n"
     "IMPORTANT OUTPUT SHAPE:\n"
     "- specialization MUST be an array of objects, not strings.\n"
     "- Each specialization item MUST be exactly {{\"t\": string, \"e\": {{\"chunk_id\": confidence}}}}.\n"
     "- INVALID (do not do): [\"keyword text\"]\n"
     "- VALID: [{{\"t\": \"explainable ai methods for safety-critical deployment\", \"e\": {{\"abc|7|1\": 0.86}}}}]\n\n"
     "Rules:\n"
     "- Use ONLY provided chunk text.\n"
     "- domain: 2-4 words.\n"
     "- specialization: 8-20 words, expertise the faculty HAS in real-world settings.\n"
     "- Focus ONLY on application intent: real-world use, deployment context, operational setting, industry/sector problems, implementation constraints, and practical impact.\n"
     "- application.domain: 3-8 items.\n"
     "- application.specialization: 4-8 items.\n"
     "- Write specialization as concise technical noun phrases (not sentences).\n"
     "- One specialization per distinct application capability; do NOT emit paraphrase variants.\n"
     "- If chunks mention any target population, sector, use-case, deployment setting, operational constraint, or practical outcome, you MUST output application keywords.\n"
     "- Return empty application only when the chunkset is purely theoretical/administrative text with no practical-use signal.\n"
     "- research must remain empty.\n"
     "- Lowercase unless proper nouns.\n"
     "- Do not invent chunk ids.\n"
     "- For specialization, e must be a non-empty object map chunk_id -> confidence in [0,1].\n"
     "- NEVER return specialization as list[str].\n"
     "- If evidence is missing for a candidate, omit it rather than returning invalid shape.\n"
     "- Domains do not require evidence ids.\n"
     "- Do not output empty keywords.\n"
     "- Keep output compact and deduplicated.\n"
     "- Confidence must be discriminative; do not assign one default value to all chunk ids."
    ),
    ("human", "Chunks JSON:\n{chunks_json}")
])

FACULTY_KEYWORD_MERGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You merge overlapping faculty keyword mentions.\n\n"
     "Input JSON schema:\n"
     "{{\n"
     "  \"research\": {{\n"
     "    \"domain\": [\"...\"],\n"
     "    \"specialization\": [{{\"t\": \"...\", \"e\": [\"chunk_id\"]}}]\n"
     "  }},\n"
     "  \"application\": {{\n"
     "    \"domain\": [\"...\"],\n"
     "    \"specialization\": [{{\"t\": \"...\", \"e\": [\"chunk_id\"]}}]\n"
     "  }}\n"
     "}}\n\n"
     "Output JSON schema:\n"
     "{{\n"
     "  \"research\": {{\n"
     "    \"domain\": [\"...\"],\n"
     "    \"specialization\": [{{\"t\": \"...\", \"e\": [\"chunk_id\"]}}]\n"
     "  }},\n"
     "  \"application\": {{\n"
     "    \"domain\": [\"...\"],\n"
     "    \"specialization\": [{{\"t\": \"...\", \"e\": [\"chunk_id\"]}}]\n"
     "  }}\n"
     "}}\n\n"
     "IMPORTANT OUTPUT SHAPE:\n"
     "- specialization MUST be an array of objects, not strings.\n"
     "- Each specialization item MUST be exactly: {{\"t\": \"...\", \"e\": [\"chunk_id\", ...]}}\n"
     "- e must contain only chunk_id strings (no confidence values).\n"
     "- INVALID (do not do): [\"keyword text\"]\n"
     "- INVALID (do not do): {{\"t\": \"...\", \"e\": {{\"chunk_id\": 0.8}}}}\n\n"
     "Rules:\n"
     "- Keep keywords within the same section (research/application).\n"
     "- Domain: merge exact duplicates and very-close duplicates.\n"
     "- Specialization: merge near-duplicates with same meaning; keep one clear wording.\n"
     "- For merged specialization rows, union evidence ids in e (deduplicated).\n"
     "- Do not move keywords between sections.\n"
     "- Preserve evidence ids exactly from input; do not invent ids.\n"
     "- Do not invent new keywords.\n"
     "- If nothing to merge, return input as-is.\n"
     "- Return only valid JSON matching the output schema."
    ),
    ("human", "Mentions JSON:\n{mentions_json}")
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
     "- specialization items for research and application\n\n"
     "Important clarification:\n"
     "- The given specialization phrases represent areas that the faculty member is professionally experienced in.\n\n"
     "Input specialization item shape:\n"
     "- {{\"t\": \"specialization text\", \"snippet_ids\": {{\"chunk_id\": confidence}}}}\n"
     "- snippet_ids are provenance only; do not alter them.\n\n"
     "Task:\n"
     "- For EACH input specialization item, output one object with:\n"
     "  - t: exact original specialization text (unchanged)\n"
     "  - w: number in [0,1] for expertise/proficiency\n\n"
     "Weight guidance:\n"
     "- 0.85–1.00: primary expertise, strong evidence in context\n"
     "- 0.60–0.84: strong working expertise\n"
     "- 0.35–0.59: some experience / occasional involvement\n"
     "- 0.10–0.34: weak/unclear evidence (use if mentioned but not supported)\n\n"
     "Rules:\n"
     "- Do NOT invent new specialization phrases.\n"
     "- Keep text exactly the same as input for t.\n"
     "- If input is non-empty, output for research/application must not be empty.\n"
     "- Output must include every input t exactly once in its same section.\n"
     "- If uncertain, assign low weight instead of dropping the item.\n"
     "- If evidence is unclear, be conservative (lower w).\n"
     "- Return ONLY JSON in the specified output format."
    ),
    ("human",
     "FACULTY CONTEXT (JSON):\n{context_json}\n\n"
     "SPECIALIZATIONS INPUT (JSON):\n{spec_json}\n\n"
     "Return weighted specializations JSON."
    )
])

FACULTY_SPECIALIZATION_WEIGHT_FLAT_PUB_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign expertise weights to faculty specialization keywords.\n\n"
     "You will receive one JSON object with:\n"
     "- faculty_context: faculty profile + flat publication strings\n"
     "- specializations: research/application arrays of items with idx and t\n\n"
     "Input item shape:\n"
     "- {{\"idx\": 0, \"t\": \"specialization text\"}}\n\n"
     "Task:\n"
     "- Return weights for each input item as:\n"
     "{{\n"
     "  \"research\": [{{\"idx\": 0, \"w\": 0.0}}],\n"
     "  \"application\": [{{\"idx\": 0, \"w\": 0.0}}]\n"
     "}}\n\n"
     "Rules:\n"
     "- Use ONLY provided input JSON.\n"
     "- Keep idx unchanged and return each input idx exactly once in its same section.\n"
     "- Do NOT invent or remove idx.\n"
     "- w must be in [0,1].\n"
     "- If uncertain, assign a lower weight instead of omitting.\n"
     "- Return ONLY JSON."
    ),
    ("human",
     "WEIGHT INPUT JSON:\n{weight_input_json}\n\n"
     "Return weighted idx JSON."
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

OPP_CATEGORY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You classify a funding opportunity into high-level grant categories.\n\n"
     "Definitions:\n"
     "- broad area grant: large domain where many project variants can fit.\n"
     "- specialized grant: niche or specific research/specialization requirement.\n"
     "- basic research: general science/research knowledge generation.\n"
     "- applied research: research/invention focused on real-world use.\n"
     "- educational: creation/improvement/evaluation of education programs.\n\n"
     "Output JSON schema:\n"
     "{{\n"
     "  \"broad_category\": \"basic_research|applied_research|educational|unclear\",\n"
     "  \"specific_categories\": [\"snake_case_code\", \"...\"]\n"
     "}}\n\n"
     "Rules:\n"
     "- Use ONLY provided opportunity context and extracted keywords.\n"
     "- specific_categories must be short snake_case codes.\n"
     "- Include one scope marker in specific_categories when possible: broad_area or specialized.\n"
     "- Add extra topic codes when clear (e.g., k12, teacher_pd, climate_resilience).\n"
     "- Deduplicate specific_categories.\n"
     "- If evidence is insufficient, set broad_category=unclear and keep categories minimal."
    ),
    ("human",
     "OPPORTUNITY CONTEXT (JSON):\n{context_json}\n\n"
     "OPPORTUNITY KEYWORDS (JSON):\n{keywords_json}\n\n"
     "Return category JSON."
    )
])

QUERY_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from a user research query for later structuring.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- Return a single JSON object with key `candidates` = list[str].\n"
     "- Include TWO kinds of candidates:\n"
     "  (A) AREA terms: 1–3 words, broad research fields or domains.\n"
     "  (B) SPECIALIZATION statements: 8–25 words describing expertise, methods, systems, or problems relevant to the query.\n"
     "- SPECIALIZATION phrases should describe capabilities the investigator WOULD NEED for the query topic.\n"
     "- Write phrases neutrally (no 'seeks', 'needs', 'requires').\n"
     "- Prefer technical and research content; avoid administrative language.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- Target ~20–60 candidates total."
    ),
    ("human", "Context (JSON):\n{context_json}")
])

QUERY_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate structured keywords from a user research query.\n\n"
     "Output must match this JSON schema:\n"
     "{{\n"
     "  \"research\": {{\"domain\": [], \"specialization\": []}},\n"
     "  \"application\": {{\"domain\": [], \"specialization\": []}}\n"
     "}}\n\n"
     "Definitions:\n"
     "- domain: 1–3 words, broad fields defining the query topic.\n"
     "- specialization: 8–25 words describing expertise, methods, or systems relevant to the query.\n\n"
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

QUERY_SPECIALIZATION_WEIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign importance weights to specialization phrases for a user research query.\n\n"
     "You will be given:\n"
     "- query context (JSON)\n"
     "- specialization lists for research and application\n\n"
     "Important clarification:\n"
     "- The given specialization phrases represent capabilities that are IMPORTANT for the query topic.\n\n"
     "Task:\n"
     "- For EACH specialization phrase, output an object with:\n"
     "  - t: the exact original specialization text (unchanged)\n"
     "  - w: a number in [0,1] representing how critical this capability is\n\n"
     "Weight guidance:\n"
     "- 0.85–1.00: core capability\n"
     "- 0.60–0.84: important capability\n"
     "- 0.35–0.59: supporting capability\n"
     "- 0.10–0.34: minor capability\n\n"
     "Rules:\n"
     "- Do NOT invent new specialization phrases.\n"
     "- Keep the text exactly the same as input for t.\n"
     "- Base weights ONLY on evidence in the provided query context.\n"
     "- If importance is unclear, be conservative (assign a lower weight).\n"
     "- Return ONLY JSON in the specified output format."
    ),
    ("human",
     "QUERY CONTEXT (JSON):\n{context_json}\n\n"
     "SPECIALIZATIONS INPUT (JSON):\n{spec_json}\n\n"
     "Return weighted specializations JSON."
    )
])
