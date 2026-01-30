from langchain_core.prompts import ChatPromptTemplate

GROUP_JUSTIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write concise, accurate justifications for a faculty TEAM matching a GRANT. "
     "Use only the provided JSON. Do not invent details."),
    ("human",
     """
Return STRICT JSON that matches the schema.

Rules:
- one_paragraph: 5-8 sentences, explain why team fits grant and how they complement each other.
- member_roles: one entry per team member, role must be short, why must be 1-2 sentences.
- coverage: list key covered areas (strong/partial) and clearly missing areas.

INPUT_JSON:
{input_json}
""".strip())
])