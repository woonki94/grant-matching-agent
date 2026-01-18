from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../root
sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
import re
from typing import List, Tuple, Dict

from langchain_openai import ChatOpenAI

from config import OPENAI_MODEL, OPENAI_API_KEY
from dao.faculty_dao import FacultyDAO
from dao.opportunity_dao import OpportunityDAO
from db.db_conn import SessionLocal
from services.keywords.generate_context import opportunity_to_keyword_context
from services.prompts.group_matching_prompt import NEEDS_PROMPT
from dto.llm_response_dto import NeedsOut
_WORD = re.compile(r"[a-z0-9]+")


def build_needs_chain():
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    return NEEDS_PROMPT | llm.with_structured_output(NeedsOut)

def extract_opportunity_needs(*, opportunity_id: str, opp_context: dict) -> NeedsOut:
    chain = build_needs_chain()
    opp_json = json.dumps(opp_context, ensure_ascii=False)
    return chain.invoke({"opportunity_id": opportunity_id, "opp_json": opp_json})



def _tokens(s: str) -> set[str]:
    return set(_WORD.findall((s or "").lower()))

def faculty_keyword_text(fac_kw: dict) -> str:
    # compress keywords into one searchable blob
    parts: List[str] = []
    for top in ("research", "application"):
        block = fac_kw.get(top) or {}
        parts.extend(block.get("domain") or [])
        parts.extend(block.get("specialization") or [])
    return " ".join(p for p in parts if isinstance(p, str))

def need_tokens(need_label: str, need_desc: str) -> set[str]:
    return _tokens(need_label) | _tokens(need_desc)

def score_faculty_for_need(fac_kw: dict, need_label: str, need_desc: str) -> float:
    """
    0..1 overlap score. Simple + stable.
    """
    text = faculty_keyword_text(fac_kw)
    if not text.strip():
        return 0.0
    fac_toks = _tokens(text)
    need_toks = need_tokens(need_label, need_desc)
    if not need_toks:
        return 0.0
    inter = fac_toks & need_toks
    return min(1.0, len(inter) / max(3, len(need_toks)))  # dampen

def build_coverage_matrix(
    faculty_rows: List[Tuple[int, str | None, str | None, dict]],
    needs: List[dict],
) -> Dict[int, Dict[str, float]]:
    """
    Returns: {faculty_id: {need_id: score}}
    faculty_rows: (faculty_id, name, email, keywords_json)
    needs: list of Need dicts
    """
    out: Dict[int, Dict[str, float]] = {}
    for fid, _, _, kw in faculty_rows:
        out[fid] = {}
        for n in needs:
            out[fid][n["need_id"]] = score_faculty_for_need(
                kw or {}, n.get("label", ""), n.get("description", "")
            )
    return out


def greedy_team_select(
    *,
    faculty_ids: List[int],
    needs: List[dict],
    coverage: Dict[int, Dict[str, float]],
    team_size: int,
) -> Tuple[List[int], List[str], List[str]]:
    """
    Returns (selected_faculty_ids, covered_need_ids, missing_need_ids)
    """
    need_weights = {n["need_id"]: int(n.get("weight", 3)) for n in needs}
    need_must = {n["need_id"]: bool(n.get("must_have", False)) for n in needs}

    selected: List[int] = []
    covered: set[str] = set()

    # Helper: value of adding faculty f given current covered set
    def marginal_gain(fid: int) -> float:
        gain = 0.0
        for nid, w in need_weights.items():
            if nid in covered:
                continue
            s = coverage.get(fid, {}).get(nid, 0.0)
            # require some minimum overlap to count
            if s >= 0.15:
                gain += w * s
        return gain

    # First, try to satisfy must-haves
    must_ids = [nid for nid in need_weights.keys() if need_must.get(nid)]
    for nid in must_ids:
        best = None
        best_s = -1.0
        for fid in faculty_ids:
            if fid in selected:
                continue
            s = coverage.get(fid, {}).get(nid, 0.0)
            if s > best_s:
                best_s = s
                best = fid
        if best is not None and best_s >= 0.15 and len(selected) < team_size:
            selected.append(best)
            # update covered using this faculty
            for nnid, s in coverage.get(best, {}).items():
                if s >= 0.15:
                    covered.add(nnid)

    # Fill remaining slots by marginal gain
    while len(selected) < team_size:
        best = None
        best_gain = 0.0
        for fid in faculty_ids:
            if fid in selected:
                continue
            g = marginal_gain(fid)
            if g > best_gain:
                best_gain = g
                best = fid

        if best is None or best_gain <= 0:
            break

        selected.append(best)
        for nnid, s in coverage.get(best, {}).items():
            if s >= 0.15:
                covered.add(nnid)

    all_need_ids = [n["need_id"] for n in needs]
    covered_ids = [nid for nid in all_need_ids if nid in covered]
    missing_ids = [nid for nid in all_need_ids if nid not in covered]
    return selected, covered_ids, missing_ids


def main(opportunity_id: str, team_size: int = 3, limit_faculty: int = 200):
    with SessionLocal() as sess:
        opp_dao = OpportunityDAO(sess)
        fac_dao = FacultyDAO(sess)

        opps = opp_dao.read_opportunities_by_ids_with_relations([opportunity_id])
        opp = opps[0] if opps else None
        if not opp:
            raise ValueError(f"Opportunity not found: {opportunity_id}")

        opp_ctx = opportunity_to_keyword_context(opp)

        needs_out = extract_opportunity_needs(
            opportunity_id=opportunity_id,
            opp_context=opp_ctx,
        )

        # Candidate pool: simplest version = first N faculty with keywords
        faculty_rows = []
        count = 0
        for fac in fac_dao.iter_faculty_with_relations(stream=False):
            if limit_faculty and count >= limit_faculty:
                break
            if not fac.keyword or not fac.keyword.keywords:
                continue
            faculty_rows.append((fac.faculty_id, fac.name, fac.email, fac.keyword.keywords))
            count += 1

        needs = [n.model_dump() for n in needs_out.needs]
        coverage = build_coverage_matrix(faculty_rows, needs)
        print(coverage)
        faculty_ids = [r[0] for r in faculty_rows]

        selected_ids, covered_ids, missing_ids = greedy_team_select(
            faculty_ids=faculty_ids,
            needs=needs,
            coverage=coverage,
            team_size=needs_out.suggested_team_size if team_size == 0 else team_size,
        )

        for n in needs_out.needs:
            mh = " (must)" if n.must_have else ""
            print(f"  - {n.need_id}: {n.label}{mh}  [w={n.weight}]")

        print(selected_ids)
        print(covered_ids),
        print(missing_ids)
    
        selected_rows = [r for r in faculty_rows if r[0] in set(selected_ids)]

        print("\n" + "=" * 90)
        print(f"Opportunity: {needs_out.opportunity_title} ({opportunity_id})")
        print(f"Scope confidence: {needs_out.scope_confidence:.2f}")
        print("Needs:")
        for n in needs_out.needs:
            mh = " (must)" if n.must_have else ""
            print(f"  - {n.need_id}: {n.label}{mh}  [w={n.weight}]")

        print("\nSelected team:")
        for fid, name, email, _ in selected_rows:
            print(f"  - {name or '<unknown>'}  <{email or 'no-email'}>  (id={fid})")

        print("\nCoverage:")
        print("  covered:", covered_ids)
        print("  missing:", missing_ids)
        print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunity-id", required=True)
    #TODO: determine teamsize by the funded amount
    parser.add_argument("--team-size", type=int, default=4, help="0 = use LLM suggested size")
    parser.add_argument("--limit-faculty", type=int, default=200)
    args = parser.parse_args()
    main(args.opportunity_id, team_size=args.team_size, limit_faculty=args.limit_faculty)