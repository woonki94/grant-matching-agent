from __future__ import annotations
from typing import Dict, List, Tuple
import math

from typing import Dict, List, Tuple
from db.db_conn import SessionLocal
from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from utils.keyword_accessor import keywords_for_matching


def solve_team_milp_soft_coverage(
    *,
    F: List[int],
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    K: int = 4,
    alpha: Dict[str, float] | None = None,
    solver_name: str = "cbc",
    msg: bool = False,
) -> Dict:
    """
    MILP:
      max sum_s alpha_s * sum_i w[s][i] * z[s][i]
      s.t. sum_f x[f] <= K
           z[s][i] >= c[f][s][i] * x[f]   for all f,s,i
           x[f] in {0,1}, z in [0,1]
    """
    alpha = alpha or {"application": 1.0, "research": 0.5}

    # ---- lazy import so your project doesn't require PuLP everywhere ----
    import pulp

    secs = ["application", "research"]
    I = {"application": I_app, "research": I_res}

    def wf(sec: str, i: int) -> float:
        return float((w.get(sec) or {}).get(i, 1.0))

    def cf(fid: int, sec: str, i: int) -> float:
        return float((((c.get(fid) or {}).get(sec) or {}).get(i, 0.0)))

    # ---- model ----
    prob = pulp.LpProblem("team_soft_coverage", pulp.LpMaximize)

    # decision vars
    x = {fid: pulp.LpVariable(f"x_{fid}", lowBound=0, upBound=1, cat=pulp.LpBinary) for fid in F}
    z = {
        (sec, i): pulp.LpVariable(f"z_{sec}_{i}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for sec in secs for i in I[sec]
    }

    # objective
    prob += pulp.lpSum(
        float(alpha.get(sec, 1.0)) * wf(sec, i) * z[(sec, i)]
        for sec in secs for i in I[sec]
    )

    # team size
    prob += pulp.lpSum(x[fid] for fid in F) <= int(K), "team_size"

    # max-linearization: z >= c * x
    # (If x=0 -> constraint is z >= 0. If x=1 -> z >= c)
    for fid in F:
        for sec in secs:
            for i in I[sec]:
                cij = cf(fid, sec, i)
                if cij <= 0.0:
                    continue
                prob += z[(sec, i)] >= cij * x[fid], f"lb_{fid}_{sec}_{i}"

        # upper bound: z cannot exceed what the selected team provides
        # this prevents solver from setting z=1 when selecting nobody
    for sec in secs:
        for i in I[sec]:
            prob += (
                z[(sec, i)]
                <= pulp.lpSum(cf(fid, sec, i) * x[fid] for fid in F),
                f"ub_{sec}_{i}",
            )

    # ---- solve ----
    if solver_name.lower() == "cbc":
        solver = pulp.PULP_CBC_CMD(msg=msg)
    elif solver_name.lower() == "glpk":
        solver = pulp.GLPK_CMD(msg=msg)
    else:
        solver = pulp.PULP_CBC_CMD(msg=msg)

    status = prob.solve(solver)

    # ---- extract ----
    status_str = pulp.LpStatus.get(prob.status, str(prob.status))
    obj = float(pulp.value(prob.objective) or 0.0)

    selected = [fid for fid in F if float(x[fid].value() or 0.0) >= 0.5]

    z_out = {sec: {i: float(z[(sec, i)].value() or 0.0) for i in I[sec]} for sec in secs}

    # useful “what contributed” view (not unique, but we can show who achieved the max)
    argmax = {sec: {} for sec in secs}
    for sec in secs:
        for i in I[sec]:
            best_f = None
            best = -1.0
            for fid in selected:
                val = cf(fid, sec, i)
                if val > best:
                    best = val
                    best_f = fid
            argmax[sec][i] = {"fid": best_f, "c": best if best > 0 else 0.0}

    return {
        "status": status_str,
        "objective": obj,
        "selected_faculty": selected,
        "z_coverage": z_out,     # soft coverage per requirement
        "argmax": argmax,        # which selected faculty most covered each requirement
    }



def requirements_indexed_text_only(kw_text: dict) -> dict:
    out = {"application": {}, "research": {}}
    for sec in ("application", "research"):
        specs = ((kw_text.get(sec) or {}).get("specialization") or [])
        for i, s in enumerate(specs):
            if isinstance(s, dict) and "t" in s:
                out[sec][str(i)] = str(s["t"])
            else:
                out[sec][str(i)] = str(s)
    return out


def requirement_weights_from_raw(kw_raw: dict) -> Dict[str, Dict[int, float]]:
    w: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
    kw_raw = kw_raw or {}
    for sec in ("application", "research"):
        specs = ((kw_raw.get(sec) or {}).get("specialization") or [])
        for i, s in enumerate(specs):
            if isinstance(s, dict):
                w[sec][i] = float(s.get("w", 1.0))
            else:
                w[sec][i] = 1.0
    return w


def build_milp_inputs_for_opportunity(
    *,
    sess,
    opportunity_id: str,
    limit_rows: int = 500,
) -> Tuple[List[int], List[int], List[int], Dict[str, Dict[int, float]], Dict[int, Dict[str, Dict[int, float]]]]:
    """
    Returns (F, I_app, I_res, w, c) for ONE opportunity.
    """
    match_dao = MatchDAO(sess)
    opp_dao = OpportunityDAO(sess)

    opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
    if not opps:
        raise ValueError(f"Opportunity not found: {opportunity_id}")
    opp = opps[0]

    # --- requirement universe + weights ---
    kw_raw = getattr(opp.keyword, "keywords", {}) or {}
    w = requirement_weights_from_raw(kw_raw)

    kw_text = keywords_for_matching(kw_raw)  # may strip weights; fine
    req_text_indexed = requirements_indexed_text_only(kw_text)

    I_app = sorted(int(k) for k in (req_text_indexed.get("application") or {}).keys())
    I_res = sorted(int(k) for k in (req_text_indexed.get("research") or {}).keys())

    # safety: fill missing weights with 1.0 (if any mismatch)
    for i in I_app:
        w["application"].setdefault(i, 1.0)
    for i in I_res:
        w["research"].setdefault(i, 1.0)

    # --- match rows => F + c ---
    match_rows = match_dao.list_matches_for_opportunity(opportunity_id, limit=limit_rows)
    if not match_rows:
        raise ValueError("No match rows found.")

    F = sorted({int(r["faculty_id"]) for r in match_rows})

    c: Dict[int, Dict[str, Dict[int, float]]] = {fid: {"application": {}, "research": {}} for fid in F}

    for r in match_rows:
        fid = int(r["faculty_id"])
        cov = r.get("covered") or {}
        for sec in ("application", "research"):
            sec_map = cov.get(sec) or {}
            for k, v in sec_map.items():
                try:
                    idx = int(k)
                    cval = float(v)
                except Exception:
                    continue
                prev = c[fid][sec].get(idx, 0.0)
                c[fid][sec][idx] = max(prev, cval)

    return F, I_app, I_res, w, c


def run_team_milp(
    opportunity_id: str,
    *,
    K: int = 4,
    limit_rows: int = 500,
):
    with SessionLocal() as sess:
        F, I_app, I_res, w, c = build_milp_inputs_for_opportunity(
            sess=sess,
            opportunity_id=opportunity_id,
            limit_rows=limit_rows,
        )

        res = solve_team_milp_soft_coverage(
            F=F,
            I_app=I_app,
            I_res=I_res,
            w=w,
            c=c,
            K=K,
            alpha={"application": 1.0, "research": 0.5},
            solver_name="cbc",
            msg=False,
        )

        print("Status:", res["status"])
        print("Objective:", res["objective"])
        print("Selected faculty:", res["selected_faculty"])

        # Optional: show top covered requirements by z
        z = res["z_coverage"]
        for sec in ("application", "research"):
            top = sorted(z[sec].items(), key=lambda kv: kv[1], reverse=True)[:10]
            print(f"\nTop z coverage ({sec}):")
            for idx, val in top:
                print(f"  idx={idx:>2} z={val:.3f}  (best_f={res['argmax'][sec][idx]['fid']}, c={res['argmax'][sec][idx]['c']:.2f})")

        return res


if __name__ == "__main__":
    opp_id = "60b8b017-30ec-4f31-a160-f00b7ee384e7"
    run_team_milp(opp_id, K=3, limit_rows=500)