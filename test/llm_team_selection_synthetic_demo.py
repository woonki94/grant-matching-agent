import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.matching.team_candidate_llm_selector import select_candidate_teams_with_llm


def scenario_high_weight_blindspot() -> Dict[str, Any]:
    requirement_weights = {
        "application": {0: 0.2, 1: 0.4, 2: 1.0},
        "research": {0: 0.2, 1: 0.4, 2: 1.0},
    }
    return {
        "name": "high_weight_blindspot",
        "desired_team_count": 2,
        "opportunity_id": "syn-opp-001",
        "description": "Candidate[0] is strong overall but Candidate[1] better covers highest-weight idx=2.",
        "candidates": [
            {
                "team": [11, 12],
                "score": 2.30,
                "final_coverage": {
                    "application": {0: 1.0, 1: 0.9, 2: 0.5},
                    "research": {0: 1.0, 1: 0.9, 2: 0.5},
                },
                "member_coverages": {
                    11: {"application": {0: 1.0, 1: 0.4, 2: 0.0}, "research": {0: 1.0, 1: 0.4, 2: 0.0}},
                    12: {"application": {0: 0.0, 1: 0.9, 2: 0.5}, "research": {0: 0.0, 1: 0.9, 2: 0.5}},
                },
            },
            {
                "team": [11, 13],
                "score": 2.25,
                "final_coverage": {
                    "application": {0: 1.0, 1: 0.6, 2: 1.0},
                    "research": {0: 1.0, 1: 0.6, 2: 1.0},
                },
                "member_coverages": {
                    11: {"application": {0: 1.0, 1: 0.4, 2: 0.0}, "research": {0: 1.0, 1: 0.4, 2: 0.0}},
                    13: {"application": {0: 0.0, 1: 0.6, 2: 1.0}, "research": {0: 0.0, 1: 0.6, 2: 1.0}},
                },
            },
            {
                "team": [12, 13],
                "score": 2.15,
                "final_coverage": {
                    "application": {0: 0.4, 1: 0.9, 2: 1.0},
                    "research": {0: 0.4, 1: 0.9, 2: 1.0},
                },
                "member_coverages": {
                    12: {"application": {0: 0.4, 1: 0.9, 2: 0.5}, "research": {0: 0.4, 1: 0.9, 2: 0.5}},
                    13: {"application": {0: 0.0, 1: 0.6, 2: 1.0}, "research": {0: 0.0, 1: 0.6, 2: 1.0}},
                },
            },
        ],
        "requirement_weights": requirement_weights,
    }


def scenario_redundancy_vs_complementarity() -> Dict[str, Any]:
    requirement_weights = {
        "application": {0: 0.5, 1: 0.8, 2: 0.9},
        "research": {0: 0.5, 1: 0.8, 2: 0.9},
    }
    return {
        "name": "redundancy_vs_complementarity",
        "desired_team_count": 2,
        "opportunity_id": "syn-opp-002",
        "description": "Top-2 by score are near-duplicates; LLM can choose one plus a complementary alternative.",
        "candidates": [
            {
                "team": [21, 22],
                "score": 2.80,
                "final_coverage": {
                    "application": {0: 0.9, 1: 0.9, 2: 0.7},
                    "research": {0: 0.9, 1: 0.9, 2: 0.7},
                },
                "member_coverages": {
                    21: {"application": {0: 0.9, 1: 0.9, 2: 0.2}, "research": {0: 0.9, 1: 0.9, 2: 0.2}},
                    22: {"application": {0: 0.8, 1: 0.8, 2: 0.7}, "research": {0: 0.8, 1: 0.8, 2: 0.7}},
                },
            },
            {
                "team": [21, 23],
                "score": 2.76,
                "final_coverage": {
                    "application": {0: 0.9, 1: 0.9, 2: 0.72},
                    "research": {0: 0.9, 1: 0.9, 2: 0.72},
                },
                "member_coverages": {
                    21: {"application": {0: 0.9, 1: 0.9, 2: 0.2}, "research": {0: 0.9, 1: 0.9, 2: 0.2}},
                    23: {"application": {0: 0.82, 1: 0.82, 2: 0.72}, "research": {0: 0.82, 1: 0.82, 2: 0.72}},
                },
            },
            {
                "team": [24, 25],
                "score": 2.60,
                "final_coverage": {
                    "application": {0: 0.6, 1: 0.8, 2: 1.0},
                    "research": {0: 0.6, 1: 0.8, 2: 1.0},
                },
                "member_coverages": {
                    24: {"application": {0: 0.6, 1: 0.2, 2: 1.0}, "research": {0: 0.6, 1: 0.2, 2: 1.0}},
                    25: {"application": {0: 0.3, 1: 0.8, 2: 0.3}, "research": {0: 0.3, 1: 0.8, 2: 0.3}},
                },
            },
        ],
        "requirement_weights": requirement_weights,
    }


def scenario_tie_break_diversity() -> Dict[str, Any]:
    requirement_weights = {
        "application": {0: 0.3, 1: 0.7, 2: 1.0},
        "research": {0: 0.3, 1: 0.7, 2: 1.0},
    }
    return {
        "name": "tie_break_diversity",
        "desired_team_count": 2,
        "opportunity_id": "syn-opp-003",
        "description": "First two candidates have same score; LLM can use diversity/complementarity as tie-break.",
        "candidates": [
            {
                "team": [31, 32],
                "score": 2.40,
                "final_coverage": {
                    "application": {0: 0.9, 1: 0.8, 2: 0.8},
                    "research": {0: 0.9, 1: 0.8, 2: 0.8},
                },
                "member_coverages": {
                    31: {"application": {0: 0.9, 1: 0.8, 2: 0.2}, "research": {0: 0.9, 1: 0.8, 2: 0.2}},
                    32: {"application": {0: 0.6, 1: 0.6, 2: 0.8}, "research": {0: 0.6, 1: 0.6, 2: 0.8}},
                },
            },
            {
                "team": [31, 33],
                "score": 2.40,
                "final_coverage": {
                    "application": {0: 0.9, 1: 0.8, 2: 0.8},
                    "research": {0: 0.9, 1: 0.8, 2: 0.8},
                },
                "member_coverages": {
                    31: {"application": {0: 0.9, 1: 0.8, 2: 0.2}, "research": {0: 0.9, 1: 0.8, 2: 0.2}},
                    33: {"application": {0: 0.2, 1: 0.5, 2: 0.8}, "research": {0: 0.2, 1: 0.5, 2: 0.8}},
                },
            },
            {
                "team": [34, 35],
                "score": 2.35,
                "final_coverage": {
                    "application": {0: 0.6, 1: 0.9, 2: 0.9},
                    "research": {0: 0.6, 1: 0.9, 2: 0.9},
                },
                "member_coverages": {
                    34: {"application": {0: 0.6, 1: 0.9, 2: 0.3}, "research": {0: 0.6, 1: 0.9, 2: 0.3}},
                    35: {"application": {0: 0.2, 1: 0.4, 2: 0.9}, "research": {0: 0.2, 1: 0.4, 2: 0.9}},
                },
            },
        ],
        "requirement_weights": requirement_weights,
    }


def run_scenario(scenario: Dict[str, Any]) -> None:
    name = scenario["name"]
    desired = scenario["desired_team_count"]
    candidates = scenario["candidates"]
    fallback_indices = list(range(min(desired, len(candidates))))

    print("\n" + "=" * 88)
    print(f"[SCENARIO] {name}")
    print(f"Description: {scenario['description']}")
    print(f"Deterministic top-{desired} baseline indices: {fallback_indices}")
    print("Candidates:")
    for i, c in enumerate(candidates):
        print(f"  idx={i} team={c['team']} score={c['score']:.4f}")

    out = select_candidate_teams_with_llm(
        opportunity_id=scenario["opportunity_id"],
        desired_team_count=desired,
        candidates=candidates,
        requirement_weights=scenario["requirement_weights"],
    )

    print("\nLLM selection result:")
    print(f"  selected_indices={out['selected_indices']}")
    print(f"  reason={out.get('reason')}")
    for i, c in enumerate(out.get("selected_candidates", [])):
        print(f"  picked[{i}] team={c.get('team')} score={c.get('score'):.4f}")


def main() -> None:
    scenarios = [
        scenario_high_weight_blindspot(),
        scenario_redundancy_vs_complementarity(),
        scenario_tie_break_diversity(),
    ]

    print("Running LLM synthetic team-selection scenarios (print-only, no asserts).")
    for s in scenarios:
        run_scenario(s)


if __name__ == "__main__":
    main()
