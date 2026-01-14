from langchain_core.prompts import ChatPromptTemplate

NEEDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract what this funding opportunity needs from a research team.\n"
     "Return ONLY JSON matching the schema.\n"
     "Rules:\n"
     "- Use ONLY the provided opportunity context.\n"
     "- If the FOA is generic (no technical topics), set scope_confidence low (<=0.4)\n"
     "  and produce broad STEM needs + proposal/process needs.\n"
     "- needs: 4–8 items total.\n"
     "- label: short; description: 1 sentence; weight 1..5.\n"
     "- must_have: true for the 1–2 most essential needs.\n"
    ),
    ("human",
     "OPPORTUNITY_ID: {opportunity_id}\n"
     "OPPORTUNITY CONTEXT (JSON):\n{opp_json}\n")
])