from langchain_core.prompts import ChatPromptTemplate

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

#TODO: let's force not to print coverage score(leave it for debugging purpose)
WHY_WORKING_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze why this team matches a grant.\n"
     "Use only provided JSON and return WhyWorkingOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- summary: concise 2-4 sentence fit summary.\n"
     "- member_strengths: one entry per faculty with up to 10 bullets.\n"
     "- Each bullet format: <grant requirement>: <why this faculty can handle it>.\n"
     "- strong/partial: concise evidence points based on weighted covered requirements.")
])

#TODO: let's force not to print coverage score(leave it for debugging purpose)
WHY_NOT_WORKING_DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze why a team may not work for a grant.\n"
     "Use only provided JSON and return WhyNotWorkingOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- why_not_working: practical risks and uncovered elements.\n"
     "- missing: explicit uncovered requirements, prioritized by importance when possible.")
])

RECOMMENDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You provide the final recommendation for a grant-team match.\n"
     "Use only provided JSON and return RecommendationOut JSON only."),
    ("user",
     "Input JSON:\n{input_json}\n\n"
     "Task:\n"
     "- Set match_quality to one of good|moderate|bad.\n"
     "- recommendation: one paragraph decision.\n"
     "- If bad, include exact phrase: Do not pursue.\n"
     "- If good/moderate, provide concrete strengthening actions.")
])
