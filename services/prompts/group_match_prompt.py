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
     "- grant_title: concise title from grant_context.title.\n"
     "- grant_link: use grant_context.opportunity_link when available.\n"
     "- grant_quick_explanation: 3-5 sentences explaining what this grant is about.\n"
     "- priority_themes: 3-6 concise bullets about high-priority themes.")
])

TEAM_ROLE_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign concise role labels for each faculty in a grant team.\n"
     "Use only provided JSON. Do not invent facts.\n"
     "Use faculty_lookup to map each faculty_name to the correct faculty_id in output.\n"
     "When evidence_text is present, ground each role in concrete evidence snippets.\n"
     "Return TeamRoleOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- For each team member, produce member_roles entry with faculty_id, role, why.\n"
     "- Role should be short and specific (e.g., 'Naval Materials Lead').\n"
     "- why should be 1 sentence grounded in provided keywords/coverage.")
])

WHY_WORKING_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze why this TEAM matches a grant with faculty-by-faculty evidence.\n"
     "Use only provided JSON. Do not invent facts.\n"
     "Use faculty_lookup to map each faculty_name to the correct faculty_id in output.\n"
     "Use evidence titles from faculty_spec_keywords when available.\n"
     "Use only provided JSON and return WhyWorkingOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- summary: 4-7 sentences with this flow:\n"
     "  1) one opening sentence on overall team fit,\n"
     "  2) one concise sentence per faculty (in faculty_lookup order) describing strongest contribution,\n"
     "  3) one closing sentence on execution readiness.\n"
     "- member_strengths: one entry per faculty in faculty_lookup.\n"
     "- For each member entry, provide up to 5 bullets.\n"
     "- Each bullet should be one sentence: contribution first, then evidence.\n"
     "- Preferred bullet pattern: <contribution>. Evidence: <title 1>; <title 2>.\n"
     "- strong: 3-8 concise team-level strengths.\n"
     "- partial: 1-5 team-level partially covered areas.\n"
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
     "- why_not_working: practical team-level improvement factors (not raw keyword dumps).\n"
     "- Explain what capability is missing and why it matters for execution.\n"
     "- missing: concise grouped capability gaps (3-6 items), avoid near-duplicate phrasing.\n"
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
     "- recommendation: one paragraph decision.\n"
     "- Encourage to pursue if the team is strong\n"
     "- Provide concrete strengthening actions when feasible.\n"
     "- If the fit is fundamentally poor, clearly advise against pursuit.\n"
     "- Do NOT mention explicit numeric weights or coverage values.")
])
