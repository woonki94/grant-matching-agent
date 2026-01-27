from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional

from dao.match_dao import MatchDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal

from utils.keyword_accessor import keywords_for_matching


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


def _get_req_text_and_weights(opp) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, float]]]:
    """
    Returns:
      req_text[sec][i] = requirement text (specialization t)
      w[sec][i] = requirement weight
    """
    kw_raw = getattr(opp.keyword, "keywords", {}) or {}

    # weights from raw (with {t,w})
    w = requirement_weights_from_raw(kw_raw)

    # text indices must match what you feed to LLM (likely string-only after normalization)
    kw_text = keywords_for_matching(kw_raw)
    req_idx_text = requirements_indexed_text_only(kw_text)

    req_text: Dict[str, Dict[int, str]] = {"application": {}, "research": {}}
    for sec in ("application", "research"):
        sec_map = req_idx_text.get(sec) or {}
        for k, t in sec_map.items():
            i = int(k)
            req_text[sec][i] = str(t)
            w[sec].setdefault(i, 1.0)  # safety

    return req_text, w


def _extract_faculty_coverage_from_match_rows(
    match_rows: List[dict],
    faculty_id: int,
) -> Dict[str, Dict[int, float]]:
    """
    Returns:
      c[sec][i] = coverage confidence for this faculty (only where present)
    Your match row covered format:
      covered: {"application": {"idx": c, ...}, "research": {"idx": c, ...}}
    """
    c: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}

    # there may be multiple rows per faculty (rare), take max per idx
    for r in match_rows:
        if int(r["faculty_id"]) != int(faculty_id):
            continue
        cov = r.get("covered") or {}
        for sec in ("application", "research"):
            sec_map = cov.get(sec) or {}
            for k, v in sec_map.items():
                try:
                    i = int(k)
                    cval = float(v)
                except Exception:
                    continue
                prev = c[sec].get(i, 0.0)
                c[sec][i] = max(prev, cval)

    return c


def print_weighted_coverage_for_first_faculty(
    opportunity_id: str,
    *,
    faculty_id: Optional[int] = None,
    limit_rows: int = 500,
    top_n: int = 50,
):
    """
    Prints per-need weighted coverage for ONE faculty on ONE opportunity.
      score = w[sec][i] * c[sec][i]
    """
    with SessionLocal() as sess:
        match_dao = MatchDAO(sess)
        opp_dao = OpportunityDAO(sess)

        opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
        if not opps:
            raise ValueError(f"Opportunity not found: {opportunity_id}")
        opp = opps[0]

        match_rows = match_dao.list_matches_for_opportunity(opportunity_id, limit=limit_rows)
        if not match_rows:
            print("No match rows found.")
            return

        # Choose "first faculty" if not provided
        F = sorted({int(r["faculty_id"]) for r in match_rows})
        if not F:
            print("No faculty IDs found in match rows.")
            return

        fid = int(faculty_id) if faculty_id is not None else F[0]

        req_text, w = _get_req_text_and_weights(opp)
        c = _extract_faculty_coverage_from_match_rows(match_rows, fid)

        print("=" * 100)
        print(f"Opportunity: {opportunity_id}")
        print(f"Faculty: {fid}")
        print("-" * 100)

        grand_total = 0.0

        for sec in ("application", "research"):
            rows = []
            for i, cval in (c.get(sec) or {}).items():
                ww = float(w[sec].get(i, 1.0))
                score = ww * float(cval)
                rows.append((score, i, ww, float(cval), req_text[sec].get(i, "")))

            rows.sort(reverse=True, key=lambda x: x[0])

            total = sum(r[0] for r in rows)
            grand_total += total

            print(f"\nSECTION: {sec.upper()}")
            print(f"  matched needs count: {len(rows)}")
            print(f"  total weighted score: {total:.4f}\n")

            # pretty print
            for (score, i, ww, cval, text) in rows[:top_n]:
                text_short = (text[:110] + "...") if len(text) > 113 else text
                print(f"  idx={i:>2} | w={ww:.2f} | c={cval:.2f} | w*c={score:.4f} | {text_short}")

            if len(rows) > top_n:
                print(f"  ... ({len(rows) - top_n} more)")

        print("\n" + "-" * 100)
        print(f"GRAND TOTAL (app+res): {grand_total:.4f}")
        print("=" * 100)


if __name__ == "__main__":
    opp_id = "60b8b017-30ec-4f31-a160-f00b7ee384e7"
    print_weighted_coverage_for_first_faculty(
        opp_id,
        faculty_id=45,   # None => first faculty in match rows
        limit_rows=500,
        top_n=50,
    )