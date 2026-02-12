from langchain_core.prompts import ChatPromptTemplate

GRANT_BRIEF_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You summarize a grant opportunity from provided JSON only.\n"
     "Do not invent facts.\n"
     "Return GrantBriefOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- grant_title: concise title.\n"
     "- grant_link: use provided link.\n"
     "- grant_quick_explanation: 3-5 sentences explaining what this grant is about.\n"
     "- priority_themes: 3-6 concise bullets about high-priority themes.")
])

TEAM_ROLE_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You assign concise role labels for each faculty in a grant team.\n"
     "Use only provided JSON. Do not invent facts.\n"
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
     "You analyze why this team matches a grant.\n"
     "Use only provided JSON and return WhyWorkingOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- summary: concise 2-4 sentence fit summary.\n"
     "- member_strengths: one entry per faculty with up to 5 bullets.\n"
     "- member_strengths MUST include one entry for every faculty_id in the team.\n"
     "- Each bullet format: <grant requirement>: <why this faculty can handle it>.\n"
     "- Do NOT mention explicit numeric weights or coverage values.\n"
     "- strong/partial: concise evidence points in plain language.")
])


WHY_NOT_WORKING_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze why a team may not work for a grant.\n"
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
