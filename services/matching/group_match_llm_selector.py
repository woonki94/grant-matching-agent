from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from config import get_llm_client
from dto.llm_response_dto import TeamCandidateSelectionOut
from logging_setup import setup_logging
from services.prompts.team_selection_prompt import TEAM_CANDIDATE_SELECTION_PROMPT

setup_logging("matching")
logger = logging.getLogger(__name__)


class GroupMatchLLMSelector:
    @staticmethod
    def _normalize_indices(indices: List[int], size: int) -> List[int]:
        out: List[int] = []
        seen = set()
        for idx in indices:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= size:
                continue
            if idx in seen:
                continue
            out.append(idx)
            seen.add(idx)
        return out

    def select_candidate_teams_with_llm(
        self,
        *,
        opportunity_id: str,
        desired_team_count: int,
        candidates: List[Dict[str, Any]],
        requirement_weights: Dict[str, Dict[int, float]],
    ) -> Dict[str, Any]:
        if desired_team_count < 1:
            raise ValueError("desired_team_count must be >= 1")
        if not candidates:
            return {
                "selected_candidates": [],
                "selected_indices": [],
                "reason": "No candidates were provided.",
            }

        desired = min(desired_team_count, len(candidates))
        report = {
            "opportunity_id": opportunity_id,
            "desired_team_count": desired,
            "requirement_weights": requirement_weights,
            "candidates": [
                {
                    "idx": i,
                    "team": c.get("team"),
                    "score": c.get("score"),
                    "final_coverage": c.get("final_coverage"),
                    "member_coverages": c.get("member_coverages"),
                }
                for i, c in enumerate(candidates)
            ],
        }
        fallback_indices = list(range(desired))

        try:
            llm = get_llm_client().build()
            chain = TEAM_CANDIDATE_SELECTION_PROMPT | llm.with_structured_output(TeamCandidateSelectionOut)
            out: TeamCandidateSelectionOut = chain.invoke(
                {"report_json": json.dumps(report, ensure_ascii=False)}
            )

            picked = self._normalize_indices(out.selected_candidate_indices, len(candidates))
            if len(picked) < desired:
                for idx in fallback_indices:
                    if idx not in picked:
                        picked.append(idx)
                    if len(picked) == desired:
                        break

            return {
                "selected_candidates": [candidates[i] for i in picked],
                "selected_indices": picked,
                "reason": out.reason,
            }
        except Exception as exc:
            logger.error(
                "LLM team candidate selection failed for opportunity_id=%s: %s",
                opportunity_id,
                exc,
            )
            return {
                "selected_candidates": [candidates[i] for i in fallback_indices],
                "selected_indices": fallback_indices,
                "reason": "Fallback to deterministic top candidates.",
            }
