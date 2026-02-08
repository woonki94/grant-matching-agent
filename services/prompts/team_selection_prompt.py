from langchain_core.prompts import ChatPromptTemplate


TEAM_CANDIDATE_SELECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are selecting final recommendations from pre-ranked candidate teams.\n"
            "Input JSON contains:\n"
            "- opportunity_id\n"
            "- desired_team_count\n"
            "- requirement_weights by section/index (importance weights)\n"
            "- candidates[] where each item has:\n"
            "  idx, team (faculty ids), score (weighted objective), final_coverage, member_coverages\n\n"
            "Decision objective:\n"
            "1) Primary: maximize weighted quality using requirement_weights and coverage.\n"
            "2) Secondary: choose a diverse set of teams (avoid near-duplicates if quality is similar).\n"
            "3) Use member_coverages to verify that each selected team has meaningful complementary roles.\n\n"
            "Weight handling:\n"
            "- Prioritize teams that cover high-weight requirements strongly.\n"
            "- Low-weight coverage should not outweigh missing high-weight coverage.\n"
            "- Use final_coverage and member_coverages together to justify weighted tradeoffs.\n\n"
            "Hard constraints:\n"
            "- Return only indices that exist in candidates[idx].\n"
            "- No duplicates in selected_candidate_indices.\n"
            "- Select exactly desired_team_count indices when possible; otherwise select as many valid as available.\n"
            "- Do not invent faculty ids, scores, or indices.\n\n"
            "Tie-break policy:\n"
            "- Prefer higher score.\n"
            "- If still tied, prefer better coverage spread across sections/indices.\n"
            "- If still tied, prefer smaller idx for deterministic output.\n\n"
            "Return concise reason referencing score/coverage/complementarity tradeoffs."
        ),
        ("user", "{report_json}"),
    ]
)
