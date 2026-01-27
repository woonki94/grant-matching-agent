from langchain_core.prompts import ChatPromptTemplate

REDUNDANCY_PAIR_PENALTY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You help select a team of faculty for a research grant.\n"
     "Your job: identify REDUNDANT PAIRS of faculty.\n"
     "Redundant means two faculty strongly cover the same grant requirements, especially high-weight requirements.\n\n"
     "You will receive JSON with:\n"
     "- grant requirements (idx, text, weight)\n"
     "- candidates with base_score and top_hits\n"
     "- current team selection and K\n\n"
     "Return ONLY JSON with pair_penalties.\n"
     "Be SPARSE: include only clearly redundant pairs.\n"
     "Penalty p is in score units (typical 0.5 to 5.0) and should be larger for overlap on high-weight requirements.\n"
     "Do not penalize complementary pairs."
    ),
    ("user", "{report_json}")
])