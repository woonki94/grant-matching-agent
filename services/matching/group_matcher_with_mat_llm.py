from __future__ import annotations

import json

from config import get_llm_client
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import PairPenaltiesOut
from services.matching.group_matcher_milp import build_milp_inputs_for_opportunity

from typing import Dict, List, Tuple, Any

from typing import Dict, List, Tuple, Any, Optional

from services.prompts.group_match_prompt import REDUNDANCY_PAIR_PENALTY_PROMPT
from utils.keyword_accessor import keywords_for_matching


def get_redundancy_penalty_chain():
    llm = get_llm_client().build()
    return REDUNDANCY_PAIR_PENALTY_PROMPT | llm.with_structured_output(PairPenaltiesOut)


def llm_pair_penalty_fn(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    report: the compact team report dict
    returns: [{"f": int, "g": int, "p": float}, ...]
    """
    chain = get_redundancy_penalty_chain()

    out: PairPenaltiesOut = chain.invoke({
        "report_json": json.dumps(report, ensure_ascii=False)
    })

    pairs = []
    for pp in (out.pair_penalties or []):
        f = int(pp.f)
        g = int(pp.g)
        if f == g:
            continue
        p = float(pp.p)
        if p <= 0:
            continue
        pairs.append({"f": f, "g": g, "p": p})
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
        s += float(alpha.get("research", 1.0))     * sum(wf("research", i)     * cf(fid, "research", i)     for i in I_res)
        scores[fid] = s
    return scores


def top_hits_for_faculty(
    fid: int,
    *,
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    alpha: Dict[str, float],
    top_m: int = 8,
    c_min: float = 0.15,
) -> List[Dict[str, Any]]:
    rows = []
    for sec, I in (("application", I_app), ("research", I_res)):
        a = float(alpha.get(sec, 1.0))
        sec_c = (c.get(fid) or {}).get(sec) or {}
        for i in I:
            cc = float(sec_c.get(i, 0.0))
            if cc < c_min:
                continue
            ww = float((w.get(sec) or {}).get(i, 1.0))
            contrib = a * ww * cc
            rows.append((contrib, sec, i, ww, cc))
    rows.sort(reverse=True, key=lambda x: x[0])
    out = [{"sec": sec, "idx": i, "w": ww, "c": cc} for _, sec, i, ww, cc in rows[:top_m]]
    return out


def build_team_report(
    *,
    candidate_fids: List[int],
    base_scores: Dict[int, float],
    team_current: List[int],
    req_text: Dict[str, Dict[int, str]],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    I_app: List[int],
    I_res: List[int],
    alpha: Dict[str, float],
    K: int,
    top_m: int = 8,
) -> Dict[str, Any]:
    # grant requirements payload (only those indices that exist in req_text)
    def req_list(sec: str) -> List[Dict[str, Any]]:
        items = []
        for i, text in (req_text.get(sec) or {}).items():
            items.append({"idx": int(i), "text": str(text), "w": float((w.get(sec) or {}).get(int(i), 1.0))})
        # sort by weight desc
        items.sort(key=lambda d: d["w"], reverse=True)
        return items

    candidates = []
    for fid in candidate_fids:
        candidates.append({
            "fid": int(fid),
            "base_score": float(base_scores.get(fid, 0.0)),
            "top_hits": top_hits_for_faculty(
                fid,
                I_app=I_app, I_res=I_res, w=w, c=c, alpha=alpha,
                top_m=top_m,
            )
        })

    # sort candidates by base score descending
    candidates.sort(key=lambda d: d["base_score"], reverse=True)

    return {
        "grant": {"requirements": {"application": req_list("application"), "research": req_list("research")}},
        "candidates": candidates,
        "team_current": [int(x) for x in team_current],
        "settings": {"K": int(K)}
    }

def redundancy_overlap_mass(
    team: List[int],
    *,
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    alpha: Dict[str, float],
) -> float:
    # average pair overlap mass
    import itertools

    def wf(sec, i): return float((w.get(sec) or {}).get(i, 1.0))
    def cf(fid, sec, i): return float((((c.get(fid) or {}).get(sec) or {}).get(i, 0.0)))

    secs = ("application", "research")
    I = {"application": I_app, "research": I_res}

    pairs = list(itertools.combinations(team, 2))
    if not pairs:
        return 0.0

    total = 0.0
    for f, g in pairs:
        ov = 0.0
        for sec in secs:
            a = float(alpha.get(sec, 1.0))
            for i in I[sec]:
                ov += a * wf(sec, i) * min(cf(f, sec, i), cf(g, sec, i))
        total += ov

    return total / len(pairs)

def is_excellent(team: List[int], *, red_score: float, red_max: float = 1.5) -> bool:
    return red_score <= red_max


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

def iterate_team_selection_with_llm_redundancy(
    *,
    F_all: List[int],
    I_app: List[int],
    I_res: List[int],
    w: Dict[str, Dict[int, float]],
    c: Dict[int, Dict[str, Dict[int, float]]],
    req_text: Dict[str, Dict[int, str]],
    K: int,
    alpha: Dict[str, float],
    topN: int = 20,
    lam_grid: List[float] = [0.0, 0.5, 1.0, 2.0, 4.0],
    max_iter: int = 3,
    red_max: float = 1.5,
    llm_pair_penalty_fn=None,  # function(report)->pair_penalties list
) -> Dict[str, Any]:
    # 1) base scores for all faculty
    base_scores = compute_base_scores(F=F_all, I_app=I_app, I_res=I_res, w=w, c=c, alpha=alpha)

    # 2) shortlist topN
    cand_sorted = sorted(F_all, key=lambda fid: base_scores.get(fid, 0.0), reverse=True)
    candidates = cand_sorted[:topN]

    # baseline team (topK by base score)
    team_current = candidates[:K]

    last = None
    for it in range(max_iter):
        # Build report and ask LLM for pair penalties
        report = build_team_report(
            candidate_fids=candidates,
            base_scores=base_scores,
            team_current=team_current,
            req_text=req_text,
            w=w, c=c,
            I_app=I_app, I_res=I_res,
            alpha=alpha,
            K=K,
        )

        if llm_pair_penalty_fn is None:
            pair_penalties = []  # fallback: no penalties
        else:
            pair_penalties = llm_pair_penalty_fn(report)

        # Sweep lambda grid and pick first "excellent" (or best objective)
        best = None
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

            sol_pack = {
                "iter": it,
                "lambda": lam,
                "team": team,
                "objective": sol["objective"],
                "status": sol["status"],
                "redundancy": red,
                "pair_penalties_used": len(pair_penalties),
            }

            # store best by objective if no excellent
            if best is None or sol_pack["objective"] > best["objective"]:
                best = sol_pack

            if is_excellent(team, red_score=red, red_max=red_max):
                return {"ok": True, "result": sol_pack, "report": report}

        # if none excellent, move to best found and iterate again (LLM sees updated team)
        last = best
        team_current = best["team"]

    return {"ok": False, "best": last}

def build_req_text_indexed(opp_keywords_raw: dict) -> Dict[str, Dict[int, str]]:
    """
    req_text[sec][idx] = specialization text
    Must match indexing used to create match_rows['covered'].
    """
    kw_text = keywords_for_matching(opp_keywords_raw)
    out: Dict[str, Dict[int, str]] = {"application": {}, "research": {}}
    for sec in ("application", "research"):
        specs = ((kw_text.get(sec) or {}).get("specialization") or [])
        for i, s in enumerate(specs):
            out[sec][i] = str(s)
    return out


def debug_team_objective_breakdown(team, base_scores, pair_penalties, lam):
    # normalize pairs
    pmap = {}
    for it in pair_penalties:
        f, g = int(it["f"]), int(it["g"])
        if f == g:
            continue
        if f > g:
            f, g = g, f
        pmap[(f, g)] = max(float(it["p"]), pmap.get((f, g), 0.0))

    base = sum(float(base_scores.get(fid, 0.0)) for fid in team)

    pairs_in_team = []
    penalty = 0.0
    team_set = set(team)
    for (f, g), p in pmap.items():
        if f in team_set and g in team_set:
            pairs_in_team.append((f, g, p))
            penalty += p

    obj = base - lam * penalty

    print("\n--- objective breakdown ---")
    print("team:", team)
    print("base:", base)
    print("penalized_pairs_in_team:", pairs_in_team)
    print("penalty_sum:", penalty)
    print("lambda:", lam)
    print("base - lambda*penalty:", obj)

if __name__ == '__main__':
    opp_id = "60b8b017-30ec-4f31-a160-f00b7ee384e7"
    K = 3
    topN = 20
    lam_grid = [0.0, 0.5, 1.0, 2.0, 4.0]

    alpha = {"application": 1.0, "research": 0.5}

    with SessionLocal() as sess:
        # --- load opp keywords for req_text ---
        opp = OpportunityDAO(sess).read_opportunities_by_ids_with_relations([opp_id])[0]
        kw_raw = getattr(opp.keyword, "keywords", {}) or {}
        req_text = build_req_text_indexed(kw_raw)

        # --- get F,I,w,c from your existing builder (ensure index alignment!) ---
        F, I_app, I_res, w, c = build_milp_inputs_for_opportunity(
            sess=sess,
            opportunity_id=opp_id,
            limit_rows=500,
        )

    # --- base scores + shortlist ---
    base_scores = compute_base_scores(F=F, I_app=I_app, I_res=I_res, w=w, c=c, alpha=alpha)
    candidates = sorted(F, key=lambda fid: base_scores.get(fid, 0.0), reverse=True)[:topN]
    team_current = candidates[:K]  # baseline team

    # --- build report + call LLM once (redundancy only) ---
    report = build_team_report(
        candidate_fids=candidates,
        base_scores=base_scores,
        team_current=team_current,
        req_text=req_text,
        w=w, c=c,
        I_app=I_app, I_res=I_res,
        alpha=alpha,
        K=K,
        top_m=8,
    )

    pair_penalties = llm_pair_penalty_fn(report)
    print("\nLLM pair penalties:")
    print(json.dumps(pair_penalties, indent=2))

    # --- sweep lambda, solve MILP, pick best by (objective) or by redundancy threshold ---
    best = None
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
        red = redundancy_overlap_mass(team, I_app=I_app, I_res=I_res, w=w, c=c, alpha=alpha)

        row = {
            "lambda": lam,
            "status": sol["status"],
            "objective": sol["objective"],
            "team": team,
            "redundancy": red,
        }
        print("\n", json.dumps(row, indent=2))

        if best is None or row["objective"] > best["objective"]:
            best = row

    print("\nBEST:")
    print(json.dumps(best, indent=2))

    debug_team_objective_breakdown(team, base_scores, pair_penalties, lam)