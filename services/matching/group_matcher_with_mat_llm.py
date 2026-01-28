from __future__ import annotations

import json
import math

from config import get_llm_client
from dao.group_match_dao import GroupMatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import PairPenaltiesOut
from services.matching.group_matcher_milp import build_milp_inputs_for_opportunity

from typing import Dict, List, Tuple, Any

from typing import Dict, List, Tuple, Any, Optional

from services.prompts.group_match_prompt import REDUNDANCY_PAIR_PENALTY_PROMPT
from utils.keyword_accessor import keywords_for_matching, build_req_text_indexed

def redundancy_penalty_fn_full(
    *,
    candidate_fids: List[int],
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    alpha: Dict[str, float],
    k: float = 0.5,              # exponential tilt strength
) -> List[Dict[str, Any]]:
    """
    Deterministic pairwise redundancy penalty (importance-aware), without normalization.

    Assumptions:
      - weights w[sec][i] are present for all i in I_sec
      - weights lie in [0,1]
      - coverage c_{f,sec,i} lies in [0,1]

    Penalty:
      p_fg = sum_sec alpha_sec * sum_i exp(-k * w_sec,i) * min(c_f, c_g)

    Raises ValueError if any required weight is missing.
    """

    if k < 0:
        raise ValueError(f"k must be >= 0. Got {k}")

    def cf(fid: int, sec: str, i: int) -> float:
        return float((((c.get(fid) or {}).get(sec) or {}).get(i, 0.0)))

    def wf(sec: str, i: int) -> float:
        if sec not in w:
            raise ValueError(f"Missing weight section: '{sec}'")
        sec_w = w[sec]
        if i not in sec_w:
            raise ValueError(f"Missing weight for section '{sec}', index: {i}")
        val = float(sec_w[i])
        # optional strict range check (uncomment if you want hard enforcement)
        # if not (0.0 <= val <= 1.0):
        #     raise ValueError(f"Weight out of [0,1] for section '{sec}', index {i}: {val}")
        return val

    F = [int(x) for x in candidate_fids]
    pairs: List[Dict[str, Any]] = []

    aa = float(alpha.get("application", 1.0))
    rr = float(alpha.get("research", 1.0))

    for a_i in range(len(F)):
        for a_j in range(a_i + 1, len(F)):
            f, g = F[a_i], F[a_j]
            p_fg = 0.0

            # application
            for i in I_app:
                w_i = wf("application", i)
                m = math.exp(-k * w_i)
                p_fg += aa * m * min(cf(f, "application", i), cf(g, "application", i))

            # research
            for i in I_res:
                w_i = wf("research", i)
                m = math.exp(-k * w_i)
                p_fg += rr * m * min(cf(f, "research", i), cf(g, "research", i))

            if p_fg > 0:
                pairs.append({"f": f, "g": g, "p": round(p_fg, 6)})

    return pairs


def compute_base_scores(
    *,
    F: List[int],
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    alpha: Dict[str, float],
) -> Dict[int, float]:
    def wf(sec: str, i: int) -> float:
        return float((w.get(sec) or {}).get(i, 1.0))
    def cf(fid: int, sec: str, i: int) -> float:
        return float((((c.get(fid) or {}).get(sec) or {}).get(i, 0.0)))

    scores: Dict[int, float] = {}
    for fid in F:
        s = 0.0
        s += float(alpha.get("application", 1.0)) * sum(wf("application", i) * cf(fid, "application", i) for i in I_app)
        s += float(alpha.get("research", 1.0))  * sum(wf("research", i)     * cf(fid, "research", i)     for i in I_res)
        scores[fid] = s
    return scores


def team_redundancy_from_pair_penalties(
    team: List[int],
    pair_penalties: List[Dict[str, Any]],
    *,
    average: bool = True,
) -> float:
    team_set = set(int(x) for x in team)

    # normalize pair map
    pmap: Dict[tuple[int, int], float] = {}
    for it in pair_penalties:
        f, g = int(it["f"]), int(it["g"])
        if f == g:
            continue
        if f > g:
            f, g = g, f
        pmap[(f, g)] = max(float(it["p"]), pmap.get((f, g), 0.0))

    # sum penalties inside team
    s = 0.0
    cnt = 0
    team_list = sorted(team_set)
    for i in range(len(team_list)):
        for j in range(i + 1, len(team_list)):
            key = (team_list[i], team_list[j])
            if key in pmap:
                s += pmap[key]
            cnt += 1

    if not average:
        return s
    return s / cnt if cnt else 0.0

def solve_team_with_pair_penalties_milp(
    *,
    candidate_fids: List[int],
    base_scores: Dict[int, float],
    pair_penalties: List[Dict[str, Any]],  # [{"f":..,"g":..,"p":..},...]
    K: int,
    lam: float,
    solver_name: str = "cbc",
    msg: bool = False,
) -> Dict[str, Any]:
    import pulp

    F = [int(x) for x in candidate_fids]
    prob = pulp.LpProblem("team_pair_penalty", pulp.LpMaximize)

    x = {fid: pulp.LpVariable(f"x_{fid}", 0, 1, cat=pulp.LpBinary) for fid in F}

    # only create y for penalized pairs
    y = {}
    P = []
    for item in pair_penalties or []:
        f = int(item["f"]); g = int(item["g"])
        if f == g:
            continue
        if f not in x or g not in x:
            continue
        if f > g:
            f, g = g, f
        key = (f, g)
        if key in y:
            continue
        p = float(item.get("p", 0.0))
        if p <= 0:
            continue
        y[key] = pulp.LpVariable(f"y_{f}_{g}", 0, 1, cat=pulp.LpBinary)
        P.append((f, g, p))

    # objective: sum base_score*x - lam*sum p*y
    prob += (
        pulp.lpSum(float(base_scores.get(fid, 0.0)) * x[fid] for fid in F)
        - float(lam) * pulp.lpSum(p * y[(f, g)] for (f, g, p) in P)
    )

    # team size
    prob += pulp.lpSum(x[fid] for fid in F) == int(K)

    # linearization y = x_f AND x_g
    for (f, g, p) in P:
        prob += y[(f, g)] <= x[f]
        prob += y[(f, g)] <= x[g]
        prob += y[(f, g)] >= x[f] + x[g] - 1

    if solver_name.lower() == "cbc":
        solver = pulp.PULP_CBC_CMD(msg=msg)
    elif solver_name.lower() == "glpk":
        solver = pulp.GLPK_CMD(msg=msg)
    else:
        solver = pulp.PULP_CBC_CMD(msg=msg)

    prob.solve(solver)

    selected = [fid for fid in F if float(x[fid].value() or 0.0) >= 0.5]
    obj = float(pulp.value(prob.objective) or 0.0)

    return {"selected": selected, "objective": obj, "status": pulp.LpStatus.get(prob.status)}


def quality_gate(
    *,
    team: List[int],
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    alpha: Dict[str, float],
    # thresholds (start low, tune later)
    min_cov: float = 0.25,
    breadth_tau: float = 0.25,
    min_breadth: float = 0.25,
    critical_w: float = 0.90,
    critical_tau: float = 0.60,
    min_critical_hit: float = 0.35,
) -> Tuple[bool, Dict[str, float]]:

    def team_max(sec: str, idx: int) -> float:
        best = 0.0
        for fid in team:
            best = max(best, float((((c.get(fid) or {}).get(sec) or {}).get(idx, 0.0))))
        return best

    def sec_metrics(sec: str, I: List[int]) -> Dict[str, float]:
        if not I:
            return {"cov": 0.0, "breadth": 0.0, "critical_hit": 1.0}

        weights = [float((w.get(sec) or {}).get(i, 1.0)) for i in I]
        denom = sum(weights) if sum(weights) > 0 else 1.0

        cov_num = 0.0
        hit = 0
        crit = 0
        crit_hit = 0

        for i in I:
            wi = float((w.get(sec) or {}).get(i, 1.0))
            mx = team_max(sec, i)

            cov_num += wi * mx
            if mx >= breadth_tau:
                hit += 1

            if wi >= critical_w:
                crit += 1
                if mx >= critical_tau:
                    crit_hit += 1

        cov = cov_num / denom
        breadth = hit / len(I)
        critical_hit = (crit_hit / crit) if crit else 1.0

        return {"cov": cov, "breadth": breadth, "critical_hit": critical_hit}

    app = sec_metrics("application", I_app)
    res = sec_metrics("research", I_res)

    a_app = float(alpha.get("application", 1.0))
    a_res = float(alpha.get("research", 1.0))
    a_sum = (a_app + a_res) if (a_app + a_res) > 0 else 1.0

    cov_total = (a_app * app["cov"] + a_res * res["cov"]) / a_sum
    breadth_total = (a_app * app["breadth"] + a_res * res["breadth"]) / a_sum
    critical_total = (a_app * app["critical_hit"] + a_res * res["critical_hit"]) / a_sum

    ok = (
        cov_total >= min_cov
        and breadth_total >= min_breadth
        and critical_total >= min_critical_hit
    )

    return ok, {
        "cov_total": cov_total,
        "breadth_total": breadth_total,
        "critical_hit_total": critical_total,
        "cov_app": app["cov"], "cov_res": res["cov"],
        "breadth_app": app["breadth"], "breadth_res": res["breadth"],
        "critical_hit_app": app["critical_hit"], "critical_hit_res": res["critical_hit"],
    }

def run_group_match_for_all_opps(
    *,
    K: int = 4,
    topN: int = 20,
    lam_grid: List[float] = [0.0, 0.5, 1.0, 2.0, 4.0],
    alpha: Dict[str, float] = {"application": 1.0, "research": 1.0},
    limit_rows: int = 500,
    batch_size: int = 200,
    commit_every: int = 25,
):
    """
    Runs group matching for all opportunities that have keywords.
    Saves one group_match_results row per (opp_id, lambda).
    """

    with SessionLocal() as sess:
        opp_dao = OpportunityDAO(sess)
        g_dao = GroupMatchDAO(sess)

        processed = 0
        saved = 0

        print(
            f"[GROUP-MATCH] start | K={K}, topN={topN}, "
            f"lambdas={lam_grid}, alpha={alpha}"
        )

        for idx, opp in enumerate(
            opp_dao.iter_opportunities_with_keywords()
        ):
            opp_id = opp.opportunity_id
            kw_raw = (opp.keyword.keywords if opp.keyword else {}) or {}

            print(f"\n[{idx:05d}] opportunity={opp_id}")

            # --- build requirement text ---
            req_text = build_req_text_indexed(kw_raw)
            if not req_text["application"] and not req_text["research"]:
                print("  ↳ skipped (no requirements)")
                continue

            # --- build inputs ---
            F, I_app, I_res, w, c = build_milp_inputs_for_opportunity(
                sess=sess,
                opportunity_id=opp_id,
                limit_rows=limit_rows,
            )
            if not F:
                print("  ↳ skipped (no faculty candidates)")
                continue

            # --- base scores + shortlist ---
            base_scores = compute_base_scores(
                F=F,
                I_app=I_app,
                I_res=I_res,
                w=w,
                c=c,
                alpha=alpha,
            )
            candidates = sorted(
                F, key=lambda fid: base_scores.get(fid, 0.0), reverse=True
            )[:topN]

            if len(candidates) < K:
                print(f"  ↳ skipped (only {len(candidates)} candidates)")
                continue

            pair_penalties = redundancy_penalty_fn_full(
                candidate_fids=candidates,
                I_app=I_app,
                I_res=I_res,
                w=w,
                c=c,
                alpha=alpha,
            )

            print(f"  penalties={len(pair_penalties)} pairs")
            print(pair_penalties)
            # --- sweep lambda ---
            for lam in lam_grid:
                sol = solve_team_with_pair_penalties_milp(
                    candidate_fids=candidates,
                    base_scores=base_scores,
                    pair_penalties=pair_penalties,
                    K=K,
                    lam=lam,
                    solver_name="cbc",
                    msg=False,
                )

                team = sol["selected"]
                red = team_redundancy_from_pair_penalties(
                    team, pair_penalties, average=True
                )

                ok, qm = quality_gate(
                    team=team,
                    I_app=I_app,
                    I_res=I_res,
                    w=w,
                    c=c,
                    alpha=alpha,
                    # tweak later; start loose
                    min_cov=0.25,
                    breadth_tau=0.25,
                    min_breadth=0.25,
                    critical_w=0.90,
                    critical_tau=0.60,
                    min_critical_hit=0.35,
                )

                if not ok:
                    print(f"    ↳ SKIP save (quality gate failed): {qm}")
                    continue

                db_row = {
                    "grant_id": opp_id,
                    "lambda": float(lam),
                    "k": int(K),
                    "top_n": int(topN),
                    "alpha": alpha,
                    "objective": float(sol.get("objective") or 0.0),
                    "redundancy": float(red),
                    "status": sol.get("status"),
                    "meta": {
                        "algo": "pair_penalty_milp",
                        "penalty_source": "deterministic_overlap",
                        "lambda_grid": lam_grid,
                        "quality": qm,
                    },
                }

                g_dao.save_group_run(db_row, team)
                saved += 1

                print(
                    f"    λ={lam:<4} team={team} "
                    f"obj={db_row['objective']:.3f} "
                    f"red={db_row['redundancy']:.3f}"
                )

            processed += 1

            # --- periodic commit ---
            if processed % commit_every == 0:
                sess.commit()
                print(
                    f"[GROUP-MATCH] committed "
                    f"(processed={processed}, saved={saved})"
                )

        sess.commit()
        print(
            f"\n[GROUP-MATCH] DONE "
            f"(processed={processed}, saved={saved})"
        )

if __name__ == '__main__':
    K = 4
    topN = 20
    lam_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    alpha = {"application": 1.0, "research": 1.0}

    run_group_match_for_all_opps(K=K, topN=topN, lam_grid=lam_grid, alpha=alpha)
