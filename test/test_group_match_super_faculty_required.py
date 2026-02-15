import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.matching.group_match_llm_selector import GroupMatchLLMSelector
from services.matching.super_faculty_selector import SuperFacultySelector

super_faculty_selector = SuperFacultySelector()
group_match_llm_selector = GroupMatchLLMSelector()


def make_synthetic_inputs():
    faculty_ids = [1, 2, 3]
    requirements = {
        # Heavier weight on keyword 2, medium on 1, light on 0.
        "application": {0: 0.1, 1: 0.5, 2: 1.0},
        "research": {0: 0.1, 1: 0.5, 2: 1.0},
    }
    coverage = {
        # Easy-to-read synthetic coverage:
        # f1 owns keyword 0, f2 owns keyword 1, f3 owns keyword 2.
        1: {"application": {0: 1.0, 1: 0.0, 2: 0.0}, "research": {0: 1.0, 1: 0.0, 2: 0.0}},
        2: {"application": {0: 0.0, 1: 1.0, 2: 0.0}, "research": {0: 0.0, 1: 1.0, 2: 0.0}},
        3: {"application": {0: 0.0, 1: 0.0, 2: 1.0}, "research": {0: 0.0, 1: 0.0, 2: 1.0}},
    }
    return faculty_ids, requirements, coverage


def test_no_required_faculty_selects_best_team():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
    )
    assert team == [2, 3]
    assert final_coverage == {"application": {0: 0.0, 1: 1.0, 2: 1.0}, "research": {0: 0.0, 1: 1.0, 2: 1.0}}


def test_single_required_faculty_is_included_and_used_as_baseline():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
        required_faculty_ids=[1],
    )
    assert 1 in team
    assert team == [1, 3]
    assert final_coverage == {"application": {0: 1.0, 1: 0.0, 2: 1.0}, "research": {0: 1.0, 1: 0.0, 2: 1.0}}


def test_single_required_faculty_with_k_one_returns_required_only():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=1,
        required_faculty_ids=[1],
    )
    assert team == [1]
    assert final_coverage == {"application": {0: 1.0, 1: 0.0, 2: 0.0}, "research": {0: 1.0, 1: 0.0, 2: 0.0}}


def test_single_required_faculty_with_duplicate_required_ids_is_deduped():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
        required_faculty_ids=[1, 1],
    )
    assert 1 in team
    assert len(team) == 2
    assert team == [1, 3]
    assert final_coverage == {"application": {0: 1.0, 1: 0.0, 2: 1.0}, "research": {0: 1.0, 1: 0.0, 2: 1.0}}


def test_multiple_required_faculty_are_all_included():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=3,
        required_faculty_ids=[1, 2],
    )
    assert sorted(team) == [1, 2, 3]
    assert final_coverage == {"application": {0: 1.0, 1: 1.0, 2: 1.0}, "research": {0: 1.0, 1: 1.0, 2: 1.0}}


def test_multiple_required_faculty_order_is_preserved_in_team_prefix():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, _ = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=3,
        required_faculty_ids=[2, 1],
    )
    assert team[:2] == [2, 1]
    assert set(team) == {1, 2, 3}


def test_multiple_required_faculty_with_k_equal_required_uses_baseline_only():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
        required_faculty_ids=[1, 2],
    )
    assert team == [1, 2]
    assert final_coverage == {"application": {0: 1.0, 1: 1.0, 2: 0.0}, "research": {0: 1.0, 1: 1.0, 2: 0.0}}


def test_k_zero_with_no_required_returns_zero_coverage():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    team, final_coverage = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=0,
    )
    assert team == []
    assert final_coverage == {"application": {0: 0.0, 1: 0.0, 2: 0.0}, "research": {0: 0.0, 1: 0.0, 2: 0.0}}


def test_missing_required_faculty_raises_value_error():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    try:
        super_faculty_selector.team_selection_super_faculty(
            cand_faculty_ids=faculty_ids,
            requirements=requirements,
            coverage=coverage,
            K=2,
            required_faculty_ids=[999],
        )
        assert False, "Expected ValueError for missing required faculty"
    except ValueError as exc:
        assert "required_faculty_ids not present in faculty_ids" in str(exc)


def test_k_less_than_number_of_required_faculty_raises_value_error():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    try:
        super_faculty_selector.team_selection_super_faculty(
            cand_faculty_ids=faculty_ids,
            requirements=requirements,
            coverage=coverage,
            K=1,
            required_faculty_ids=[1, 2],
        )
        assert False, "Expected ValueError when K is smaller than required set"
    except ValueError as exc:
        assert "K must be >=" in str(exc)


def test_returns_top_n_candidates_sorted_by_weighted_score():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    candidates = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
        num_candidates=2,
    )

    assert len(candidates) == 2
    assert candidates[0]["team"] == [2, 3]
    assert candidates[1]["team"] == [1, 3]
    assert candidates[0]["score"] > candidates[1]["score"]


def test_returns_top_n_candidates_with_required_faculty():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    candidates = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
        required_faculty_ids=[1],
        num_candidates=2,
    )

    assert len(candidates) == 2
    assert candidates[0]["team"] == [1, 3]
    assert candidates[1]["team"] == [1, 2]


def test_llm_selection_print_only_demo():
    faculty_ids, requirements, coverage = make_synthetic_inputs()
    candidates = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=2,
        num_candidates=3,
    )

    llm_candidates = []
    for cand in candidates:
        team_ids = list(cand["team"])
        llm_candidates.append(
            {
                **cand,
                "member_coverages": {
                    int(fid): coverage.get(int(fid), {"application": {}, "research": {}})
                    for fid in team_ids
                },
            }
        )

    result = group_match_llm_selector.select_candidate_teams_with_llm(
        opportunity_id="synthetic-opp-001",
        desired_team_count=2,
        candidates=llm_candidates,
        requirement_weights=requirements,
    )

    print("\n[LLM DEMO] Input candidates:")
    for idx, cand in enumerate(llm_candidates):
        print(f"  idx={idx} team={cand['team']} score={cand['score']:.4f}")

    print("\n[LLM DEMO] Selection result:")
    print(f"  selected_indices={result['selected_indices']}")
    print(f"  reason={result['reason']}")
    for i, cand in enumerate(result["selected_candidates"]):
        print(f"  picked[{i}] team={cand['team']} score={cand['score']:.4f}")


def main():
    tests = [
        test_no_required_faculty_selects_best_team,
        test_single_required_faculty_is_included_and_used_as_baseline,
        test_single_required_faculty_with_k_one_returns_required_only,
        test_single_required_faculty_with_duplicate_required_ids_is_deduped,
        test_multiple_required_faculty_are_all_included,
        test_multiple_required_faculty_order_is_preserved_in_team_prefix,
        test_multiple_required_faculty_with_k_equal_required_uses_baseline_only,
        test_k_zero_with_no_required_returns_zero_coverage,
        test_missing_required_faculty_raises_value_error,
        test_k_less_than_number_of_required_faculty_raises_value_error,
        test_returns_top_n_candidates_sorted_by_weighted_score,
        test_returns_top_n_candidates_with_required_faculty,
    ]

    for test_fn in tests:
        test_fn()
        print(f"[PASS] {test_fn.__name__}")

    # Print-only LLM selection demo on synthetic data (no assertions).
    test_llm_selection_print_only_demo()

    print(f"All tests passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
