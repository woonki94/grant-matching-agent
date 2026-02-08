import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.matching.group_match_super_faculty import run_group_match


SYNTHETIC_EMAIL_TO_ID = {
    "alice@osu.edu": 101,
    "bob@osu.edu": 102,
}

SYNTHETIC_OPP_INPUTS = {
    "syn-opp-001": {
        "F": [101, 102, 201, 202, 203],
        "w": {
            "application": {0: 0.2, 1: 0.4, 2: 1.0},
            "research": {0: 0.2, 1: 0.4, 2: 1.0},
        },
        "c": {
            101: {"application": {0: 1.0, 1: 0.1, 2: 0.0}, "research": {0: 1.0, 1: 0.1, 2: 0.0}},
            102: {"application": {0: 0.2, 1: 1.0, 2: 0.0}, "research": {0: 0.2, 1: 1.0, 2: 0.0}},
            201: {"application": {0: 0.0, 1: 0.4, 2: 1.0}, "research": {0: 0.0, 1: 0.4, 2: 1.0}},
            202: {"application": {0: 0.4, 1: 0.8, 2: 0.6}, "research": {0: 0.4, 1: 0.8, 2: 0.6}},
            203: {"application": {0: 0.7, 1: 0.2, 2: 0.9}, "research": {0: 0.7, 1: 0.2, 2: 0.9}},
        },
    },
    "syn-opp-002": {
        "F": [101, 102, 301, 302, 303],
        "w": {
            "application": {0: 0.3, 1: 0.8, 2: 0.9},
            "research": {0: 0.3, 1: 0.8, 2: 0.9},
        },
        "c": {
            101: {"application": {0: 0.9, 1: 0.3, 2: 0.0}, "research": {0: 0.9, 1: 0.3, 2: 0.0}},
            102: {"application": {0: 0.2, 1: 0.9, 2: 0.0}, "research": {0: 0.2, 1: 0.9, 2: 0.0}},
            301: {"application": {0: 0.8, 1: 0.8, 2: 0.6}, "research": {0: 0.8, 1: 0.8, 2: 0.6}},
            302: {"application": {0: 0.5, 1: 0.7, 2: 1.0}, "research": {0: 0.5, 1: 0.7, 2: 1.0}},
            303: {"application": {0: 0.3, 1: 0.9, 2: 0.8}, "research": {0: 0.3, 1: 0.9, 2: 0.8}},
        },
    },
}


class _FakeSessionCtx:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeFacultyDAO:
    def __init__(self, _sess):
        pass

    def get_faculty_id_by_email(self, email: str) -> int:
        if email not in SYNTHETIC_EMAIL_TO_ID:
            raise ValueError(f"Unknown synthetic email: {email}")
        return SYNTHETIC_EMAIL_TO_ID[email]


class _FakeMatchDAO:
    def __init__(self, _sess):
        pass

    def get_grant_ids_for_faculty(self, faculty_id: int):
        # If opp_ids is not explicitly passed, use all synthetic opportunities.
        _ = faculty_id
        return list(SYNTHETIC_OPP_INPUTS.keys())


def _fake_build_inputs_for_opportunity(*, sess, opportunity_id: str, limit_rows: int = 500):
    _ = sess
    _ = limit_rows
    row = SYNTHETIC_OPP_INPUTS[opportunity_id]
    f = list(row["F"])
    w = row["w"]
    c = row["c"]
    i_app = sorted(w["application"].keys())
    i_res = sorted(w["research"].keys())
    return f, i_app, i_res, w, c


def _team_list(rows):
    return [r["team"] for r in rows]


def main():
    faculty_emails = ["alice@osu.edu", "bob@osu.edu"]
    opp_ids = list(SYNTHETIC_OPP_INPUTS.keys())
    desired_team_count = 2

    with patch("services.matching.group_match_super_faculty.SessionLocal", _FakeSessionCtx), patch(
        "services.matching.group_match_super_faculty.FacultyDAO", _FakeFacultyDAO
    ), patch("services.matching.group_match_super_faculty.MatchDAO", _FakeMatchDAO), patch(
        "services.matching.group_match_super_faculty.build_inputs_for_opportunity",
        _fake_build_inputs_for_opportunity,
    ):
        deterministic = run_group_match(
            faculty_emails=faculty_emails,
            team_size=3,
            num_candidates=5,
            opp_ids=opp_ids,
            use_llm_selection=False,
            desired_team_count=desired_team_count,
        )
        llm_based = run_group_match(
            faculty_emails=faculty_emails,
            team_size=3,
            num_candidates=5,
            opp_ids=opp_ids,
            use_llm_selection=True,
            desired_team_count=desired_team_count,
        )

    print("Synthetic comparison using run_group_match (deterministic vs LLM).\n")
    for det, llm in zip(deterministic, llm_based):
        opp_id = det["opp_id"]
        det_top = _team_list(det["selected_teams"][:desired_team_count])
        llm_top = _team_list(llm["selected_teams"])
        print(f"Opportunity: {opp_id}")
        print(f"  Deterministic top-{desired_team_count}: {det_top}")
        print(f"  LLM selected top-{desired_team_count}: {llm_top}")
        print(f"  LLM reason: {None if llm['llm_selection'] is None else llm['llm_selection'].get('reason')}")
        print(f"  Different? {det_top != llm_top}\n")


if __name__ == "__main__":
    main()
