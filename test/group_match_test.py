from services.matching.super_faculty_selector import SuperFacultySelector
from services.matching.group_matcher_with_mat_llm import team_redundancy_from_pair_penalties, quality_gate, \
    solve_team_with_pair_penalties_milp, redundancy_penalty_fn_full, compute_base_scores

super_faculty_selector = SuperFacultySelector()


def make_dummy_inputs():
    # Faculty
    F = [1, 2, 3, 4, 5, 6,7,8,9]

    I_app = list(range(3))
    I_res = list(range(3))

    #synthetic grant keyword weights
    w = {
        "application": {0: 0.3, 1: 0.6, 2: 0.8},
        "research": {0: 0.3, 1: 0.6, 2: 0.8},
    }

    #synthetic faculty coverages to each grant keywords.
    c = {
        1: {"application": {0: 0.8, 1: 0.1, 2: 0.1}, "research": {0: 0.8, 1: 0.1, 2: 0.1}},
        2: {"application": {0: 0.8, 1: 0.1, 2: 0.1}, "research": {0: 0.8, 1: 0.1, 2: 0.1}},
        3: {"application": {0: 0.8, 1: 0.1, 2: 0.1}, "research": {0: 0.8, 1: 0.1, 2: 0.1}},

        4: {"application": {0: 0.1, 1: 0.9, 2: 0.1}, "research": {0: 0.1, 1: 0.9, 2: 0.1}},
        5: {"application": {0: 0.1, 1: 0.9, 2: 0.1}, "research": {0: 0.1, 1: 0.9, 2: 0.1}},
        6: {"application": {0: 0.1, 1: 0.9, 2: 0.1}, "research": {0: 0.1, 1: 0.9, 2: 0.1}},

        7: {"application": {0: 0.1, 1: 0.1, 2: 0.6}, "research": {0: 0.1, 1: 0.1, 2: 0.6}},
        8: {"application": {0: 0.1, 1: 0.1, 2: 0.6}, "research": {0: 0.1, 1: 0.1, 2: 0.6}},
        9: {"application": {0: 0.1, 1: 0.1, 2: 0.6}, "research": {0: 0.1, 1: 0.1, 2: 0.6}},
    }
    '''
    c = {
        1: {"application": {0: 0.6, 1: 0.6, 2: 0.6}, "research": {0: 0.6, 1: 0.6, 2: 0.6}},
        2: {"application": {0: 0.6, 1: 0.6, 2: 0.6}, "research": {0: 0.6, 1: 0.6, 2: 0.6}},
        3: {"application": {0: 0.6, 1: 0.6, 2: 0.6}, "research": {0: 0.6, 1: 0.6, 2: 0.6}},

        4: {"application": {0: 0.1, 1: 0.9, 2: 0.1}, "research": {0: 0.1, 1: 0.9, 2: 0.1}},
        5: {"application": {0: 0.1, 1: 0.9, 2: 0.1}, "research": {0: 0.1, 1: 0.9, 2: 0.1}},
        6: {"application": {0: 0.1, 1: 0.9, 2: 0.1}, "research": {0: 0.1, 1: 0.9, 2: 0.1}},

        7: {"application": {0: 0.1, 1: 0.1, 2: 0.9}, "research": {0: 0.1, 1: 0.1, 2: 0.9}},
        8: {"application": {0: 0.1, 1: 0.1, 2: 0.9}, "research": {0: 0.1, 1: 0.1, 2: 0.9}},
        9: {"application": {0: 0.1, 1: 0.1, 2: 0.9}, "research": {0: 0.1, 1: 0.1, 2: 0.9}},
    }
   '''

    alpha = {"application": 1.0, "research": 1.0}
    return F, I_app, I_res, w, c, alpha


def main():
    F, I_app, I_res, w, c, alpha = make_dummy_inputs()

    base_scores = compute_base_scores(F=F, I_app=I_app, I_res=I_res, w=w, c=c, alpha=alpha)
    candidates = sorted(F, key=lambda fid: base_scores.get(fid, 0.0), reverse=True)

    print("\nBase scores (sorted):")
    for fid in candidates:
        print(f"  fid={fid} score={base_scores[fid]:.4f}")


    pair_penalties = redundancy_penalty_fn_full(
        candidate_fids=candidates,
        I_app=I_app,
        I_res=I_res,
        w=w,
        c=c,
        alpha=alpha,
        k=5,
    )

    print(f"\nPair penalties: {len(pair_penalties)} pairs")
    for it in sorted(pair_penalties, key=lambda x: -x["p"])[:10]:
        print(f"  ({it['f']},{it['g']}) p={it['p']}")

    team_size = 4
    lam_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print("\nResults:")
    for lam in lam_grid:
        sol = solve_team_with_pair_penalties_milp(
            candidate_fids=candidates,
            base_scores=base_scores,
            pair_penalties=pair_penalties,
            K=team_size,
            lam=lam,
            solver_name="cbc",
            msg=False,
        )
        team = sol["selected"]
        red = team_redundancy_from_pair_penalties(team, pair_penalties, average=True)
        ok, qm = quality_gate(team=team, I_app=I_app, I_res=I_res, w=w, c=c, alpha=alpha)

        print(
            f"  λ={lam:<4} status={sol['status']:<10} "
            f"team={team} obj={sol['objective']:.4f} red={red:.4f} "
            f"quality_ok={ok}"
        )

    print("\nDone.")


# ----------------------------
# Synthetic test data
# ----------------------------
def build_synthetic_grants():
    """
    Returns multiple synthetic grants.
    Each grant is (requirements, coverage).
    """

    faculty_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    requirements = {
        "application": {0: 0.3, 1: 0.6, 2: 0.8},
        "research": {0: 0.3, 1: 0.6, 2: 0.8},
    }

    coverage = {
        1: {"application": {0: 0.6, 1: 0.6, 2: 0.6}, "research": {0: 0.6, 1: 0.6, 2: 0.6}},
        2: {"application": {0: 0.6, 1: 0.6, 2: 0.6}, "research": {0: 0.6, 1: 0.6, 2: 0.6}},
        3: {"application": {0: 0.6, 1: 0.6, 2: 0.6}, "research": {0: 0.6, 1: 0.6, 2: 0.6}},

        4: {"application": {0: 0.1, 1: 0.8, 2: 0.1}, "research": {0: 0.1, 1: 0.8, 2: 0.1}},
        5: {"application": {0: 0.1, 1: 0.8, 2: 0.1}, "research": {0: 0.1, 1: 0.8, 2: 0.1}},
        6: {"application": {0: 0.1, 1: 0.8, 2: 0.1}, "research": {0: 0.1, 1: 0.8, 2: 0.1}},

        7: {"application": {0: 0.1, 1: 0.1, 2: 0.9}, "research": {0: 0.1, 1: 0.1, 2: 0.9}},
        8: {"application": {0: 0.1, 1: 0.1, 2: 0.9}, "research": {0: 0.1, 1: 0.1, 2: 0.9}},
        9: {"application": {0: 0.3, 1: 0.1, 2: 0.9}, "research": {0: 0.1, 1: 0.1, 2: 0.9}},
    }

    '''
    faculty_ids = [1, 2, 3]

    requirements = {
        "application": {0:1},
        "research": {0:1},
    }

    coverage = {
        1: {"application": {0: 1.0}, "research": {0: 0.0}},
        2: {"application": {0: 0.0}, "research": {0: 1.0}},
        3: {"application": {0: 0.6}, "research": {0: 0.6}},
    }

    '''
    return faculty_ids, requirements, coverage


if __name__ == "__main__":
    #main()
    faculty_ids, requirements, coverage = build_synthetic_grants()

    K = 2
    team, final_cov = super_faculty_selector.team_selection_super_faculty(
        cand_faculty_ids=faculty_ids,
        requirements=requirements,
        coverage=coverage,
        K=K,
    )

    print("\nSelected team:", team)
    print("Final coverage:")
    for sec in final_cov:
        print(f"  {sec}: {final_cov[sec]}")
