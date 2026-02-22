from langchain_core.prompts import ChatPromptTemplate



FACULTY_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from faculty context for later structuring.\n\n"
     "You must follow a 4-step internal workflow:\n"
     "1) SCAN: identify research areas, methods, systems, problems, datasets, domains, and applications explicitly present.\n"
     "2) DRAFT: propose candidate phrases (AREA + SPECIALIZATION).\n"
     "3) FILTER: remove admin/affiliation content, vague claims, future-tense, and duplicates.\n"
     "4) NORMALIZE: lowercase unless proper nouns; keep phrases compact and technical.\n\n"
     "Output:\n"
     "- Return ONLY a single JSON object: {\"candidates\": [\"...\", ...]}.\n\n"
     "Candidate types (MUST include both):\n"
     "A) AREA terms (1–3 words): broad research fields/domains.\n"
     "B) SPECIALIZATION statements (8–25 words): specific expertise, methods, systems, or problems the faculty CURRENTLY works on.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- SPECIALIZATION must describe capabilities the faculty HOLDS (no 'seeks', 'interested in', 'aims to', 'would like').\n"
     "- Prefer technical/research content; avoid titles, departments, awards, teaching, service, hiring, outreach, funding admin.\n"
     "- Avoid generic fillers (e.g., 'machine learning techniques', 'data analysis', 'research experience').\n"
     "- Make SPECIALIZATION statements concrete: include method + object/system + purpose when possible.\n"
     "- Deduplicate aggressively (merge near-duplicates; keep the most specific one).\n\n"
     "Coverage targets:\n"
     "- Total candidates: 30–80.\n"
     "- AREA: 10–25 items.\n"
     "- SPECIALIZATION: 20–55 items.\n"
     "- If context is short, still try to produce at least 12 total candidates by using only what is explicitly stated.\n\n"
     "Return only JSON."
    ),
    ("human", "Context (JSON):\n{context_json}")
])

OPP_CANDIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract candidate keyword phrases from a funding opportunity context for later structuring.\n\n"
     "Use a 4-step internal workflow:\n"
     "1) SCAN: identify target topics, technical scope, methods, required capabilities, deliverables, and application areas.\n"
     "2) DRAFT: propose candidate phrases (AREA + SPECIALIZATION).\n"
     "3) FILTER: remove submission/eligibility/admin language and duplicates.\n"
     "4) NORMALIZE: lowercase unless proper nouns; keep phrases compact and technical.\n\n"
     "Output:\n"
     "- Return ONLY a single JSON object: {\"candidates\": [\"...\", ...]}.\n\n"
     "Candidate types (MUST include both):\n"
     "A) AREA terms (1–3 words): broad research/technical areas targeted.\n"
     "B) SPECIALIZATION statements (8–25 words): capabilities, methods, or expertise the project EXPECTS investigators to have.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context.\n"
     "- SPECIALIZATION must describe what the grant NEEDS investigators to have (expected capabilities/experience).\n"
     "- Exclude admin/process/eligibility content (deadlines, page limits, budgeting, submission portals, institutional eligibility).\n"
     "- Avoid generic fillers ('innovative solutions', 'cutting-edge', 'state-of-the-art') unless tied to specific technical content.\n"
     "- Make SPECIALIZATION statements concrete: capability + technical object + goal/deliverable when possible.\n"
     "- Deduplicate aggressively; keep the most specific version.\n\n"
     "Coverage targets:\n"
     "- Total candidates: 30–80.\n"
     "- AREA: 10–25 items.\n"
     "- SPECIALIZATION: 20–55 items.\n"
     "- If context is short, still try to produce at least 12 total candidates using only explicit text.\n\n"
     "Return only JSON."
    ),
    ("human", "Context (JSON):\n{context_json}")
])

FACULTY_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate structured keywords from faculty context.\n\n"
     "Output must match this JSON schema:\n"
     "{\n"
     "  \"research\": {\"domain\": [], \"specialization\": []},\n"
     "  \"application\": {\"domain\": [], \"specialization\": []}\n"
     "}\n\n"
     "Definitions:\n"
     "- domain: 1–3 words, broad fields defining the faculty’s work.\n"
     "- specialization: 8–25 words describing expertise, methods, or systems the faculty HAS.\n\n"
     "Agentic selection workflow (internal):\n"
     "1) SELECT: pick the best candidates that are strongly supported by the context.\n"
     "2) SORT: prefer specificity and technical clarity.\n"
     "3) BALANCE: ensure coverage across methods/systems/problems where present.\n"
     "4) VERIFY: remove anything not supported or redundant.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context and candidate phrases.\n"
     "- Do NOT invent new topics or restate candidates with new meaning.\n"
     "- You may lightly edit candidates ONLY to:\n"
     "  (a) fix casing, (b) remove trailing filler words, (c) merge duplicates, while keeping meaning unchanged.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate across all four lists (avoid repeating the same concept in multiple buckets unless clearly different).\n"
     "- Avoid administrative or institutional language.\n\n"
     "Bucket guidance:\n"
     "- research.*: fundamental methods, theory, models, systems, measurement, algorithms, architectures.\n"
     "- application.*: domains of use (health, energy, security, education, robotics, climate, etc.) and applied problem settings.\n\n"
     "Count targets:\n"
     "- research.domain: 4–10\n"
     "- research.specialization: 5–15\n"
     "- application.domain: 4–10\n"
     "- application.specialization: 5–15\n\n"
     "Return only JSON."
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
     "{\n"
     "  \"research\": {\"domain\": [], \"specialization\": []},\n"
     "  \"application\": {\"domain\": [], \"specialization\": []}\n"
     "}\n\n"
     "Definitions:\n"
     "- domain: 1–3 words, broad research or technical areas.\n"
     "- specialization: 8–25 words describing expertise, methods, or capabilities the project EXPECTS investigators to have.\n\n"
     "Agentic selection workflow (internal):\n"
     "1) SELECT: choose candidates that represent core technical scope/requirements.\n"
     "2) PRIORITIZE: prefer concrete capabilities and technical specificity.\n"
     "3) BALANCE: avoid over-indexing on one buzzword; reflect the opportunity’s breadth.\n"
     "4) VERIFY: drop anything that is admin/eligibility/process.\n\n"
     "Rules:\n"
     "- Use ONLY the provided context and candidate phrases.\n"
     "- Do NOT invent requirements not present in the text.\n"
     "- You may lightly edit candidates ONLY to:\n"
     "  (a) fix casing, (b) remove trailing filler words, (c) merge duplicates, while keeping meaning unchanged.\n"
     "- Lowercase unless proper nouns.\n"
     "- Deduplicate.\n"
     "- Avoid generic submission/eligibility language.\n\n"
     "Bucket guidance:\n"
     "- research.*: technical scope, methods, systems, measurement, models, algorithms, scientific areas.\n"
     "- application.*: mission area, impact domain, end-use setting, operational environment.\n\n"
     "Count targets:\n"
     "- research.domain: 4–10\n"
     "- research.specialization: 5–15\n"
     "- application.domain: 4–10\n"
     "- application.specialization: 5–15\n\n"
     "Return only JSON."
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

