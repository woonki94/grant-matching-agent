from langchain_core.prompts import ChatPromptTemplate

GRANT_BRIEF_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You summarize a grant opportunity from provided grant_context JSON only.\n"
     "Do not invent facts.\n"
     "Return GrantBriefOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- Read grant data from the grant_context object.\n"
     "- grant_quick_explanation: 3-5 concise sentences explaining what this grant is about.\n"
     "- Focus on scope, goals, and expected work; avoid boilerplate.")
])

TEAM_ROLE_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign concise role labels for each faculty in a grant team.\n"
     "Use only provided JSON. Do not invent facts.\n"
     "Return roles in the same order as faculty_lookup.\n"
     "Return TeamRoleOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- Produce roles: [string, ...], one role per faculty in faculty_lookup order.\n"
     "- Role should be short and specific (e.g., 'Naval Materials Lead').\n"
     "- Do NOT output faculty_id or explanation text.")
])

WHY_WORKING_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze why this TEAM matches a grant with faculty-by-faculty evidence.\n"
     "Use only provided JSON. Do not invent facts.\n"
     "Use evidence titles from faculty_spec_keywords when available.\n"
     "Use only provided JSON and return WhyWorkingOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- summary: concise 3-5 sentences on why this team can work.\n"
     "- member_strengths: one entry per faculty in faculty_lookup.\n"
     "- Each member_strengths entry must include faculty_name and bullets.\n"
     "- For each member, provide 2-5 concise bullets.\n"
     "- Each bullet should be one sentence with this format:\n"
     "  <contribution>. Evidence: <exact evidence title> [source=<publication|additional_info|attachment>].\n"
     "- For publication evidence, include year in the title when available.\n"
     "- Use exact evidence titles from input when present; do not invent titles.\n"
     "- Do NOT mention explicit numeric weights or coverage values.\n"
     "- Keep language practical and proposal-oriented.")
])


WHY_NOT_WORKING_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze why a team may not work for a grant.\n"
     "Use evidence_text when present to identify concrete missing capabilities.\n"
     "Use only provided JSON and return WhyNotWorkingOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- why_not_working: concise practical risk bullets (3-5 items), not keyword dumps.\n"
     "- Each bullet should explain one major gap and why it matters.\n"
     "- missing: concise grouped gap labels (0-3 items), avoid near-duplicates.\n"
     "- Do NOT mention explicit numeric weights or coverage values.")
])

RECOMMENDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You provide the final recommendation for a grant-team match.\n"
     "Ground decisions in the provided evidence-derived analysis.\n"
     "Use only provided JSON and return RecommendationOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- recommendation: concise decision in 2-4 sentences.\n"
     "- team_grant_fit: numeric fit score on 0.00-1.00 scale (higher is better).\n"
     "- Encourage pursuit if fit is strong.\n"
     "- If not, give only the most important strengthening actions.\n"
     "- If fit is fundamentally poor, clearly advise against pursuit.\n"
     "- Do NOT mention explicit numeric weights or coverage values.")
])
