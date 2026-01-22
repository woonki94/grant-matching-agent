from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal

# your existing normalizer (likely strips weights -> strings)
from utils.keyword_accessor import keywords_for_matching


def requirements_indexed_text_only(kw_text: dict) -> dict:
    """
    kw_text is a normalized keyword dict (specialization likely strings).
    Returns:
      {
        "application": {"0": "...", "1": "..."},
        "research":    {"0": "...", "1": "..."}
      }
    """
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
    """
    kw_raw is the DB JSON with weighted specialization items like {"t":..., "w":...}.
    Returns:
      weights = {"application":{idx:w}, "research":{idx:w}}
    """
    w: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
    kw_raw = kw_raw or {}

    for sec in ("application", "research"):
        specs = ((kw_raw.get(sec) or {}).get("specialization") or [])
        for i, s in enumerate(specs):
            if isinstance(s, dict):
                w[sec][i] = float(s.get("w", 1.0))
            else:
                # if unweighted legacy data exists, default 1.0
                w[sec][i] = 1.0
    return w


def _get_req_universe_and_weights(opp) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, float]]]:
    """
    Returns:
      reqs    = {"application":[0..], "research":[0..]}  # index universe
      weights = {"application":{idx:w}, "research":{idx:w}}
    """
    kw_raw = getattr(opp.keyword, "keywords", {}) or {}

    # (1) weights MUST come from raw keywords (before normalization)
    weights = requirement_weights_from_raw(kw_raw)

    # (2) requirement universe (indices) MUST match the text list you indexed for the LLM
    kw_text = keywords_for_matching(kw_raw)  # ok if this strips weights
    req_idx_text = requirements_indexed_text_only(kw_text)

    reqs: Dict[str, List[int]] = {"application": [], "research": []}
    for sec in ("application", "research"):
        sec_map = req_idx_text.get(sec) or {}
        idxs = sorted(int(k) for k in sec_map.keys())
        reqs[sec] = idxs

        # safety: if weights length mismatch (shouldn't), fill missing indices with 1.0
        for i in idxs:
            weights[sec].setdefault(i, 1.0)

    return reqs, weights


def print_milp_vars_for_opportunity(
    *,
    opportunity_id: str,
    tau: float = 0.8,
    limit_rows: int = 500,
):
    """
    Prints sets/params and which MILP variables would be created given your match rows.

    Assumes match_rows contain:
      - faculty_id
      - covered: {"application": {"idx": c, ...}, "research": {"idx": c, ...}}
    """
    with SessionLocal() as sess:
        match_dao = MatchDAO(sess)
        opp_dao = OpportunityDAO(sess)

        opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
        if not opps:
            raise ValueError(f"Opportunity not found: {opportunity_id}")
        opp = opps[0]

        # Requirement universe + weights from opp keywords
        reqs, w = _get_req_universe_and_weights(opp)
        I_app, I_res = reqs["application"], reqs["research"]

        # Load match rows
        match_rows = match_dao.list_matches_for_opportunity(opportunity_id, limit=limit_rows)
        if not match_rows:
            print("No match rows found.")
            return

        # Candidate faculty set
        F = sorted({int(r["faculty_id"]) for r in match_rows})

        # Build c[f][sec][idx] from covered dicts
        c: Dict[int, Dict[str, Dict[int, float]]] = {fid: {"application": {}, "research": {}} for fid in F}
        for r in match_rows:
            fid = int(r["faculty_id"])
            cov = r.get("covered") or {}
            for sec in ("application", "research"):
                sec_map = cov.get(sec) or {}
                # covered format: {"idx": cfloat}
                for k, v in sec_map.items():
                    try:
                        i = int(k)
                        c[fid][sec][i] = max(c[fid][sec].get(i, 0.0), float(v))
                    except Exception:
                        continue

        # Determine which y variables exist (edges) based on tau
        y_edges: List[Tuple[int, str, int]] = []
        for fid in F:
            for sec, I in (("application", I_app), ("research", I_res)):
                for i in I:
                    if c[fid][sec].get(i, 0.0) >= tau:
                        y_edges.append((fid, sec, i))

        # ---- PRINT ----
        print("=" * 90)
        print(f"Opportunity: {opportunity_id}")
        print(f"tau (quality threshold): {tau}")
        print("-" * 90)

        print("SETS")
        print(f"  F (faculty ids) size={len(F)}: {F[:30]}{' ...' if len(F) > 30 else ''}")
        print(f"  I_application size={len(I_app)}: {I_app}")
        print(f"  I_research    size={len(I_res)}: {I_res}")

        print("\nPARAMETERS (sample)")
        print("  w_application (idx -> weight):", {i: w["application"].get(i, 1.0) for i in I_app})
        print("  w_research    (idx -> weight):", {i: w["research"].get(i, 1.0) for i in I_res})

        sample_f = F[:5]
        for fid in sample_f:
            print(f"\n  c[f={fid}]['application'] (idx->c): {dict(sorted(c[fid]['application'].items()))}")
            print(f"  c[f={fid}]['research']    (idx->c): {dict(sorted(c[fid]['research'].items()))}")

        print("\nDECISION VARIABLES")
        print(f"  x_f: one binary per faculty => {len(F)} variables")
        print(f"  z_s_i: one binary per requirement per section => {len(I_app) + len(I_res)} variables")
        print(f"  y_f_s_i: one binary per eligible edge (c >= tau) => {len(y_edges)} variables")

        print("\nY EDGE LIST (first 50)")
        for (fid, sec, i) in y_edges[:50]:
            print(f"  y[{fid},{sec},{i}]  (c={c[fid][sec].get(i, 0.0):.2f}, w={w[sec].get(i, 1.0):.2f})")
        if len(y_edges) > 50:
            print(f"  ... ({len(y_edges) - 50} more)")

        print("\nSUGGESTED BUILD ARRAYS")
        print("  edge_score(fid,sec,idx) = w[sec][idx] * c[fid][sec][idx] (only if c>=tau)")
        print("=" * 90)


if __name__ == "__main__":

    opp_id = '60b8b017-30ec-4f31-a160-f00b7ee384e7'

    print_milp_vars_for_opportunity(
        opportunity_id=opp_id,
        tau=0.8,
        limit_rows=500,
    )