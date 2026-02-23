from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Tuple, Union

from logging_setup import setup_logging

setup_logging("matching")
logger = logging.getLogger(__name__)


class SuperFacultySelector:
    def team_selection_super_faculty(
        self,
        *,
        cand_faculty_ids: List[int],
        requirements: Dict[str, Dict[int, float]],
        coverage: Dict[int, Dict[str, Dict[int, float]]],
        K: int,
        required_faculty_ids: List[int] | None = None,
        num_candidates: int = 1,
    ) -> Union[
        Tuple[List[int], Dict[str, Dict[int, float]]],
        List[Dict[str, object]],
    ]:
        if K < 0:
            raise ValueError("K must be >= 0")
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if K > len(cand_faculty_ids):
            K = min(K, len(cand_faculty_ids))

        required_faculty_ids = required_faculty_ids or []
        missing_required = [f for f in required_faculty_ids if f not in cand_faculty_ids]
        if missing_required:
            raise ValueError(f"required_faculty_ids not present in faculty_ids: {missing_required}")
        if len(set(required_faculty_ids)) > K:
            raise ValueError("K must be >= number of required_faculty_ids")

        required_team = list(dict.fromkeys(required_faculty_ids))
        remaining_k = K - len(required_team)
        candidate_pool = [f for f in cand_faculty_ids if f not in set(required_team)]

        if remaining_k < 0:
            raise ValueError("K must be >= number of unique required_faculty_ids")
        if remaining_k > len(candidate_pool):
            remaining_k = len(candidate_pool)

        def score(covered: Dict[str, Dict[int, float]]) -> float:
            total = 0.0
            for sec in covered:
                for i in covered[sec]:
                    try:
                        weight = float(requirements[sec][i])
                    except KeyError:
                        logger.error(
                            "Missing requirement weight for section=%s index=%s while scoring team.",
                            sec,
                            i,
                        )
                        raise ValueError(
                            f"Missing requirement weight for section='{sec}', index={i}"
                        )
                    total += weight * covered[sec][i]
            return total

        candidates: List[Dict[str, object]] = []
        for extra_team in combinations(candidate_pool, remaining_k):
            team = required_team + list(extra_team)
            covered = {sec: {i: 0.0 for i in req.keys()} for sec, req in requirements.items()}

            for f in required_team:
                for sec, req in requirements.items():
                    f_sec = coverage[f][sec]
                    for i in req.keys():
                        val = f_sec.get(i, 0.0)
                        if val > covered[sec][i]:
                            covered[sec][i] = val

            for f in extra_team:
                for sec, req in requirements.items():
                    f_sec = coverage[f][sec]
                    for i in req.keys():
                        val = f_sec.get(i, 0.0)
                        if val > covered[sec][i]:
                            covered[sec][i] = val

            team_score = score(covered)
            candidates.append(
                {
                    "team": list(team),
                    "final_coverage": covered,
                    "score": team_score,
                }
            )

        candidates.sort(key=lambda x: (-float(x["score"]), list(x["team"])))
        top_candidates = candidates[:num_candidates]
        if num_candidates == 1:
            top = top_candidates[0]
            return top["team"], top["final_coverage"]
        return top_candidates
