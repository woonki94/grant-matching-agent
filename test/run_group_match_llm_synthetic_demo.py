import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.matching.team_grant_matcher import TeamGrantMatcherService


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
    return f, w, c


class _FakeContextGenerator:
    def build_matching_inputs_for_opportunity(self, *, sess, opportunity_id: str, limit_rows: int = 500):
        return _fake_build_inputs_for_opportunity(
            sess=sess,
            opportunity_id=opportunity_id,
            limit_rows=limit_rows,
        )


def _teams_by_opp(rows):
    out = {}
    for r in rows:
        oid = r["opp_id"]
        out.setdefault(oid, []).append(r["team"])
    return out


def main():
    faculty_emails = ["alice@osu.edu", "bob@osu.edu"]
    opp_ids = list(SYNTHETIC_OPP_INPUTS.keys())
    desired_team_count = 2

    with patch("services.matching.team_grant_matcher.SessionLocal", _FakeSessionCtx), patch(
        "services.matching.team_grant_matcher.FacultyDAO", _FakeFacultyDAO
    ), patch("services.matching.team_grant_matcher.MatchDAO", _FakeMatchDAO), patch(
        "services.matching.team_grant_matcher.ContextGenerator",
        _FakeContextGenerator,
    ):
        service = TeamGrantMatcherService()
        deterministic = service.run_group_match(
            faculty_emails=faculty_emails,
            team_size=3,
            num_candidates=5,
            opp_ids=opp_ids,
            use_llm_selection=False,
            desired_team_count=desired_team_count,
            group_by_opp=False,
        )
        llm_based = service.run_group_match(
            faculty_emails=faculty_emails,
            team_size=3,
            num_candidates=5,
            opp_ids=opp_ids,
            use_llm_selection=True,
            desired_team_count=desired_team_count,
            group_by_opp=False,
        )

    print("Synthetic comparison using TeamGrantMatcherService.run_group_match (deterministic vs LLM).\n")
    det_map = _teams_by_opp(deterministic)
    llm_map = _teams_by_opp(llm_based)

    for opp_id in opp_ids:
        det_top = det_map.get(opp_id, [])
        llm_top = llm_map.get(opp_id, [])
        print(f"Opportunity: {opp_id}")
        print(f"  Deterministic top-{desired_team_count}: {det_top}")
        print(f"  LLM selected top-{desired_team_count}: {llm_top}")
        print(f"  Different? {det_top != llm_top}\n")


if __name__ == "__main__":
    main()
