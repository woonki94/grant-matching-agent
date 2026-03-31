from __future__ import annotations

from typing import Any, Dict, List, Tuple

from dao.match_dao import MatchDAO


def norm(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def parse_team_ids(raw: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            fid = int(token)
        except Exception:
            continue
        if fid in seen:
            continue
        seen.add(fid)
        out.append(fid)
    return out


def build_top_rows_for_faculty(
    *,
    sess,
    faculty_id: int,
    k: int,
    opportunity_id: str | None = None,
) -> List[Tuple[str, float, float]]:
    mdao = MatchDAO(sess)

    if opportunity_id:
        oid = norm(opportunity_id)
        row = mdao.get_match_for_faculty_opportunity(
            faculty_id=int(faculty_id),
            opportunity_id=oid,
        )
        if not row:
            raise ValueError(
                f"No match row found for faculty_id={faculty_id}, opportunity_id={oid}"
            )
        return [
            (
                str(oid),
                safe_float(row.get("domain_score")),
                safe_float(row.get("llm_score")),
            )
        ]

    rows = mdao.top_matches_for_faculty(faculty_id=int(faculty_id), k=max(1, int(k)))
    out: List[Tuple[str, float, float]] = []
    for oid, domain_score, llm_score in list(rows or [])[: max(1, int(k))]:
        norm_oid = norm(oid)
        if not norm_oid:
            continue
        out.append((norm_oid, safe_float(domain_score), safe_float(llm_score)))

    if not out:
        raise ValueError(f"No matches found for faculty_id={faculty_id}")
    return out


def merge_member_coverages(member_coverages: Dict[int, Dict[str, Dict[int, float]]]) -> Dict[str, Dict[int, float]]:
    merged: Dict[str, Dict[int, float]] = {"application": {}, "research": {}}
    for _, cov in list((member_coverages or {}).items()):
        for section in ("application", "research"):
            sec_cov = dict((cov or {}).get(section) or {})
            for idx, val in sec_cov.items():
                try:
                    key = int(idx)
                    score = float(val)
                except Exception:
                    continue
                merged[section][key] = max(float(merged[section].get(key, 0.0)), score)
    return merged
