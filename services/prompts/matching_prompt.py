from langchain_core.prompts import ChatPromptTemplate

MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You score whether a faculty member is a good match for a funding opportunity.\n"
     "Return JSON with llm_score (0..1) and reason (ONE sentence <=25 words).\n"
     "Use ONLY the provided contexts.\n"),
    ("human",
     "FACULTY CONTEXT (JSON):\n{faculty_json}\n\n"
     "OPPORTUNITY CONTEXT (JSON):\n{opp_json}\n")
])
