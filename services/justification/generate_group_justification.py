from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_llm_client
from dao.opportunity_dao import OpportunityDAO
from dao.faculty_dao import FacultyDAO
from db.db_conn import SessionLocal
from dto.llm_response_dto import (
    GroupJustificationOut,
    TeamRoleOut,
    WhyWorkingOut,
    WhyNotWorkingOut,
    RecommendationOut,
)
from services.matching.group_match_super_faculty import run_group_match
from services.prompts.group_match_prompt import (
    TEAM_ROLE_DECIDER_PROMPT,
    WHY_WORKING_DECIDER_PROMPT,
    WHY_NOT_WORKING_DECIDER_PROMPT,
    RECOMMENDER_PROMPT,
)


# -------------------------
# Context building utilities
# -------------------------

def build_base_payload(
    *,
    opp_ctx: Dict[str, Any],
    fac_ctxs: List[Dict[str, Any]],
    coverage: Any,
    group_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Base payload sent to planner/writer/critic."""
    payload: Dict[str, Any] = {
        "grant": {
            "id": opp_ctx.get("opportunity_id") or opp_ctx.get("id"),
            "title": opp_ctx.get("title"),
            "agency": opp_ctx.get("agency"),
            "summary": opp_ctx.get("summary"),
            "keywords": opp_ctx.get("keywords"),
        },
        "team": [
            {
                "faculty_id": f.get("faculty_id") or f.get("id"),
                "name": f.get("name"),
                "email": f.get("email"),
                "keywords": f.get("keywords"),
                # Optional extra fields can be merged in later
            }
            for f in fac_ctxs
        ],
        "coverage": coverage,
    }
    if group_meta:
        payload["group_match"] = group_meta
    return payload


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def extract_requirement_specs(opp_ctx: Dict[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Extract requirement text/weight by section/index from opportunity keyword payload.
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {"application": {}, "research": {}}
    kw = (opp_ctx.get("keywords") or {}) if isinstance(opp_ctx, dict) else {}

    for sec in ("application", "research"):
        sec_obj = kw.get(sec) if isinstance(kw, dict) else None
        if not isinstance(sec_obj, dict):
            continue
        specs = sec_obj.get("specialization")
        if not isinstance(specs, list):
            continue
        for i, item in enumerate(specs):
            if not isinstance(item, dict):
                continue
            out[sec][i] = {
                "text": str(item.get("t") or f"{sec} requirement {i}"),
                "weight": float(item.get("w") or 0.0),
            }
    return out


# -------------------------
# Provider-agnostic LLM chain builder
# -------------------------

@dataclass
class GroupJustificationLLMs:
    writer: Any


def build_llms() -> GroupJustificationLLMs:
    """
    Uses your existing get_llm_client().build().
    For best results:
      - writer: moderate temp
    If your builder supports params, set them there; otherwise it's fine as-is.
    """
    base = get_llm_client().build()
    # If your client supports .bind(temperature=...), uncomment and tune:
    # writer = base.bind(temperature=0.4)
    writer = base
    return GroupJustificationLLMs(writer=writer)


# -------------------------
# Optional: deterministic enrichment hooks
# -------------------------

def enrich_opportunity_context(odao: OpportunityDAO, opp_ctx: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Attempt to add extra fields if your DAO/context supports them.
    Safe: will only add keys that already exist in returned context.
    """
    # If you have richer DAO methods, call them here.
    # For now, we re-fetch the same context and keep only requested extra fields if present.
    full = odao.get_opportunity_context(opp_ctx.get("opportunity_id") or opp_ctx.get("id")) or {}
    for k in fields:
        if k in full and k not in opp_ctx:
            opp_ctx[k] = full[k]
    return opp_ctx


def enrich_faculty_context(fdao: FacultyDAO, fac_ctx: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Attempt to add extra fields if your DAO/context supports them.
    If you later add methods like get_faculty_profile/get_recent_pubs, wire them here.
    """
    fid = fac_ctx.get("faculty_id") or fac_ctx.get("id")
    # Re-fetch keyword context (or swap to richer DAO calls later).
    full = fdao.get_faculty_keyword_context(fid) or {}
    # Merge requested fields if present
    for k in fields:
        if k in full and k not in fac_ctx:
            fac_ctx[k] = full[k]
    return fac_ctx


# -------------------------
# Agentic justification pipeline
# -------------------------

class JustificationEngine:
    def __init__(self, *, odao: OpportunityDAO, fdao: FacultyDAO):
        self.odao = odao
        self.fdao = fdao
        self.llms = build_llms()

        self.team_role_chain = TEAM_ROLE_DECIDER_PROMPT | self.llms.writer.with_structured_output(TeamRoleOut)
        self.why_working_chain = WHY_WORKING_DECIDER_PROMPT | self.llms.writer.with_structured_output(WhyWorkingOut)
        self.why_not_working_chain = WHY_NOT_WORKING_DECIDER_PROMPT | self.llms.writer.with_structured_output(WhyNotWorkingOut)
        self.recommender_chain = RECOMMENDER_PROMPT | self.llms.writer.with_structured_output(RecommendationOut)

    def run_one(
        self,
        *,
        opp_ctx: Dict[str, Any],
        fac_ctxs: List[Dict[str, Any]],
        coverage: Any,
        group_meta: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GroupJustificationOut, Dict[str, Any]]:
        """
        Returns (justification, trace).
        trace includes planner/critic outputs and the final context used.
        """
        trace = trace or {}
        trace.setdefault("steps", {})
        trace.setdefault("context_versions", [])

        # Build initial context
        context = build_base_payload(
            opp_ctx=opp_ctx,
            fac_ctxs=fac_ctxs,
            coverage=coverage,
            group_meta=group_meta,
        )
        trace["context_versions"].append(json.loads(safe_json(context)))

        split_input = {
            "grant": context.get("grant"),
            "team": context.get("team"),
            "coverage": context.get("coverage"),
            "requirements": extract_requirement_specs(opp_ctx),
        }
        try:
            team_roles = self.team_role_chain.invoke({"input_json": safe_json(split_input)})
            trace["steps"]["team_roles"] = team_roles.model_dump()

            why_working = self.why_working_chain.invoke({"input_json": safe_json(split_input)})
            trace["steps"]["why_working"] = why_working.model_dump()

            why_not = self.why_not_working_chain.invoke({"input_json": safe_json(split_input)})
            trace["steps"]["why_not_working"] = why_not.model_dump()

            rec_input = {
                **split_input,
                "team_roles": team_roles.model_dump(),
                "why_working": why_working.model_dump(),
                "why_not_working": why_not.model_dump(),
            }
            recommendation = self.recommender_chain.invoke({"input_json": safe_json(rec_input)})
            trace["steps"]["recommendation"] = recommendation.model_dump()

            justification = GroupJustificationOut(
                match_quality=recommendation.match_quality,
                one_paragraph=why_working.summary or "",
                member_roles=team_roles.member_roles,
                coverage={
                    "strong": why_working.strong,
                    "partial": why_working.partial,
                    "missing": why_not.missing,
                },
                member_strengths=why_working.member_strengths,
                why_not_working=why_not.why_not_working,
                recommendation=recommendation.recommendation,
            )
            return justification, trace
        except Exception:
            fallback_roles = []
            for tm in context.get("team", []) or []:
                fid = tm.get("faculty_id")
                if isinstance(fid, int):
                    fallback_roles.append(
                        {
                            "faculty_id": fid,
                            "role": "Contributor",
                            "why": "Insufficient LLM output; role inferred as general contributor.",
                        }
                    )
            justification = GroupJustificationOut(
                match_quality="moderate",
                one_paragraph="Insufficient model output to generate a complete structured justification.",
                member_roles=fallback_roles,
                coverage={"strong": [], "partial": [], "missing": []},
                member_strengths=[],
                why_not_working=["Structured writer stages failed; verify model/service health and rerun."],
                recommendation="Review this result manually and rerun generation.",
            )
            return justification, trace


def _render_markdown_report(results: List[Dict[str, Any]]) -> str:
    def _quality_from_item(item: Dict[str, Any]) -> str:
        just = item.get("justification") or {}
        q = (just.get("match_quality") or "").strip().lower()
        if q in {"good", "moderate", "bad"}:
            return q
        score = item.get("score")
        if isinstance(score, (int, float)):
            if score >= 70.0:
                return "good"
            if score >= 40.0:
                return "moderate"
        return "bad"

    def _heading_pair(quality: str) -> Tuple[str, str]:
        if quality == "good":
            return "## Why This Is a Good Match", "## Why It Might Not Work"
        if quality == "moderate":
            return "## Why This Match Might Work", "## Why This Match Likely Won't Work"
        return "## ⚠️ Critical Mismatch Alert", "## Why This Match Doesn't Work"

    def _deterministic_gap_items(item: Dict[str, Any]) -> List[str]:
        gap_rows: List[Tuple[float, str]] = []
        final_cov = item.get("final_coverage") or {}
        req_specs = item.get("requirement_specs") or {}

        for sec in ("application", "research"):
            sec_cov = final_cov.get(sec) if isinstance(final_cov, dict) else {}
            if not isinstance(sec_cov, dict):
                continue
            sec_specs = req_specs.get(sec) if isinstance(req_specs, dict) else {}
            for k, v in sec_cov.items():
                try:
                    idx = int(k)
                    cov = float(v)
                except Exception:
                    continue
                if cov > 0.05:
                    continue
                spec = sec_specs.get(idx) if isinstance(sec_specs, dict) else None
                if isinstance(spec, dict):
                    txt = str(spec.get("text") or f"{sec} requirement {idx}")
                    w = float(spec.get("weight") or 0.0)
                    gap_rows.append((w, txt))
                else:
                    gap_rows.append((0.0, f"{sec} capability gap"))

        gap_rows.sort(key=lambda x: x[0], reverse=True)
        raw = [row[1] for row in gap_rows]
        # Deduplicate by normalized text and keep concise grouped factors.
        seen = set()
        grouped: List[str] = []
        for txt in raw:
            key = txt.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            grouped.append(
                f"Strengthen capability in '{txt}' so the team can execute this requirement reliably."
            )
            if len(grouped) >= 5:
                break
        return grouped

    def _format_strength_bullet(text: str) -> str:
        s = (text or "").strip()
        if not s:
            return s
        if ":" in s:
            return s
        if " - " in s:
            left, right = s.split(" - ", 1)
            return f"{left.strip()}: {right.strip()}"
        return f"Grant requirement alignment: {s}"

    lines: List[str] = []
    for r in results:
        title = r.get("grant_title") or r.get("grant_id")
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Grant Link:** {r.get('grant_link')}")
        score = r.get("score")
        lines.append(f"**Match Score:** {score}/100" if score is not None else "**Match Score:** N/A/100")
        lines.append("")
        lines.append("---")
        lines.append("")

        if r.get("error"):
            lines.append("## ⚠️ Processing Error")
            lines.append("")
            lines.append(r["error"])
            lines.append("")
            lines.append("## Recommendation")
            lines.append("Review this opportunity manually or rerun justification generation.")
            lines.append("")
            continue

        quality = _quality_from_item(r)
        positive_heading, risk_heading = _heading_pair(quality)
        just = r.get("justification", {}) or {}
        role_by_faculty: Dict[int, str] = {}
        for mr in just.get("member_roles", []) or []:
            try:
                fid = int(mr.get("faculty_id"))
            except Exception:
                continue
            role_txt = str(mr.get("role") or "").strip()
            if role_txt:
                role_by_faculty[fid] = role_txt

        lines.append("## Team")
        lines.append("")
        for m in r.get("team_members", []):
            name = m.get("faculty_name") or f"Faculty {m.get('faculty_id')}"
            email = m.get("faculty_email")
            fid = m.get("faculty_id")
            role = role_by_faculty.get(int(fid), "Contributor") if isinstance(fid, int) else "Contributor"
            if email:
                lines.append(f"- **{name}** ({email}) — {role}")
            else:
                lines.append(f"- **{name}** — {role}")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append(positive_heading)
        lines.append("")
        strengths = just.get("member_strengths") or []
        strengths_by_faculty: Dict[int, List[str]] = {}
        for s in strengths:
            try:
                fid = int(s.get("faculty_id"))
            except Exception:
                continue
            bullets = s.get("bullets") or []
            if isinstance(bullets, list):
                strengths_by_faculty[fid] = [str(b).strip() for b in bullets if str(b).strip()]

        if quality == "bad":
            lines.append("- Overall evidence indicates a fundamental mismatch with grant requirements.")
        else:
            for m in r.get("team_members", []):
                fid = m.get("faculty_id")
                name = m.get("faculty_name") or f"Faculty {fid}"
                bullets = strengths_by_faculty.get(int(fid)) if isinstance(fid, int) else None
                lines.append(f"**{name}'s Strengths:**")
                if bullets:
                    for b in bullets[:10]:
                        lines.append(f"- {_format_strength_bullet(b)}")
                else:
                    lines.append("- Not enough faculty-specific evidence was extracted; regenerate this section.")
                lines.append("")

        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append(risk_heading)
        lines.append("")
        lines.append("**Critical Gaps:**")
        why_not = just.get("why_not_working") or []
        coverage = just.get("coverage", {}) or {}
        missing = coverage.get("missing", []) or []
        gap_items = []
        if isinstance(why_not, list):
            gap_items.extend([str(x).strip() for x in why_not if str(x).strip()])
        if isinstance(missing, list):
            gap_items.extend([str(x).strip() for x in missing if str(x).strip()])
        gap_items.extend(_deterministic_gap_items(r))
        # De-duplicate while preserving order.
        if gap_items:
            seen = set()
            deduped: List[str] = []
            for g in gap_items:
                key = g.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(g)
            gap_items = deduped
        if gap_items:
            for item in gap_items:
                lines.append(f"- {item}")
        else:
            lines.append("- No explicit missing coverage was flagged.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        recommendation = (just.get("recommendation") or "").strip()
        if quality == "bad":
            lines.append("Do not pursue.")
        elif recommendation:
            lines.append(recommendation)
        else:
            if gap_items:
                top = "; ".join(gap_items[:3])
                lines.append(
                    f"Proceed only if you add collaborators to cover the highest-priority uncovered areas: {top}. "
                    "Update scope to these requirements and rerun matching before submission."
                )
            else:
                lines.append("Refine team composition or scope based on the listed critical gaps, then reassess.")

        lines.append("")

    return "\n".join(lines).strip()

def _write_markdown_report(markdown_text: str, output_path: Optional[str] = None) -> Path:
    if output_path:
        out = Path(output_path).expanduser()
    else:
        out = PROJECT_ROOT / "outputs" / "justification_reports" / f"group_justification_{int(time.time())}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown_text, encoding="utf-8")
    return out


def run_justifications_from_group_results_agentic(
    *,
    faculty_emails: str,
    team_size: int,
    opp_ids: Optional[List[str]] = None,
    limit_rows: int = 500,
    include_trace: bool = False,
) -> str:
    def _expand_group_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize run_group_match outputs into flat rows:
        {opp_id, team, final_coverage, score}
        Supports both grouped output (selected_teams per opp) and legacy flat rows.
        """
        expanded: List[Dict[str, Any]] = []
        for r in rows:
            if "selected_teams" in r:
                opp_id = r.get("opp_id") or r.get("grant_id")
                for cand in r.get("selected_teams") or []:
                    expanded.append(
                        {
                            "opp_id": opp_id,
                            "team": cand.get("team"),
                            "final_coverage": cand.get("final_coverage"),
                            "score": cand.get("score"),
                        }
                    )
            else:
                expanded.append(
                    {
                        "opp_id": r.get("opp_id") or r.get("grant_id"),
                        "team": r.get("team"),
                        "final_coverage": r.get("final_coverage"),
                        "score": r.get("score"),
                    }
                )
        return expanded

    with SessionLocal() as sess:
        odao = OpportunityDAO(sess)
        fdao = FacultyDAO(sess)
        engine = JustificationEngine(odao=odao, fdao=fdao)

        group_results = run_group_match(
            faculty_emails=[faculty_emails],
            team_size=team_size,
            limit_rows=limit_rows,
            opp_ids=opp_ids,
        )
        normalized_rows = _expand_group_results(group_results)
        if not normalized_rows:
            raise ValueError(f"No group matches found for {faculty_emails}")

        # Caches
        opp_cache: Dict[str, Dict[str, Any]] = {}
        fac_cache: Dict[int, Dict[str, Any]] = {}

        results: List[Dict[str, Any]] = []

        for idx, r in enumerate(normalized_rows):
            opp_id = r["opp_id"]
            team: List[int] = r["team"]
            coverage = r.get("final_coverage")

            # --- opportunity context ---
            if opp_id not in opp_cache:
                opp_cache[opp_id] = odao.get_opportunity_context(opp_id) or {}
            opp_ctx = opp_cache[opp_id]

            if not opp_ctx:
                results.append({
                    "index": idx,
                    "grant_id": opp_id,
                    "grant_title": None,
                    "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                    "team": team,
                    "error": "Opportunity not found",
                })
                continue

            # --- faculty contexts ---
            fac_ctxs: List[Dict[str, Any]] = []
            for fid in team:
                if fid not in fac_cache:
                    fac_cache[fid] = fdao.get_faculty_keyword_context(fid) or {}
                if fac_cache[fid]:
                    fac_ctxs.append(fac_cache[fid])

            if not fac_ctxs:
                results.append({
                    "index": idx,
                    "grant_id": opp_id,
                    "grant_title": opp_ctx.get("title") or opp_ctx.get("opportunity_title"),
                    "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                    "team": team,
                    "error": "No faculty contexts found for team",
                })
                continue

            # Optional group metadata if you have it in r; safe default:
            group_meta = {
                "group_id": r.get("group_id") or r.get("id"),
                "lambda": r.get("lambda"),
                "k": r.get("k"),
                "objective": r.get("objective"),
                "redundancy": r.get("redundancy"),
                "meta": r.get("meta"),
            }

            try:
                justification, trace = engine.run_one(
                    opp_ctx=dict(opp_ctx),          # copy so enrichment doesn’t poison cache
                    fac_ctxs=[dict(f) for f in fac_ctxs],
                    coverage=coverage,
                    group_meta=group_meta,
                    trace={"index": idx, "opp_id": opp_id, "team": team},
                )

                out = {
                    "index": idx,
                    "grant_id": opp_id,
                    "grant_title": opp_ctx.get("title") or opp_ctx.get("opportunity_title"),
                    "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                    "team": team,
                    "team_members": [
                        {
                            "faculty_id": f.get("faculty_id") or f.get("id"),
                            "faculty_name": f.get("name"),
                            "faculty_email": f.get("email"),
                        }
                        for f in fac_ctxs
                    ],
                    "score": r.get("score"),
                    "final_coverage": coverage,
                    "requirement_specs": extract_requirement_specs(opp_ctx),
                    "justification": justification.model_dump(),
                }
                if include_trace:
                    out["trace"] = trace

                results.append(out)

            except Exception as e:
                results.append({
                    "index": idx,
                    "grant_id": opp_id,
                    "grant_title": opp_ctx.get("title") or opp_ctx.get("opportunity_title"),
                    "grant_link": f"https://simpler.grants.gov/opportunity/{opp_id}",
                    "team": team,
                    "error": f"{type(e).__name__}: {e}",
                })

        return _render_markdown_report(results)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run justification agent")
    parser.add_argument(
        "--email",
        required=True,
        help="Faculty email (e.g. abbasiB@oregonstate.edu)",
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=3,
        help="Number of agents in the team (default: 3)",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=200,
        help="Max number of rows to process",
    )
    parser.add_argument(
        "--opp-id",
        type=str,
        help="Single target opportunity id",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Output markdown file path (default: auto-generated under outputs/justification_reports)",
    )
    parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Include agent trace output",
    )

    args = parser.parse_args()

    rendered = run_justifications_from_group_results_agentic(
        faculty_emails=args.email,
        team_size=args.team_size,
        opp_ids=[args.opp_id] if args.opp_id else None,
        limit_rows=args.limit_rows,
        include_trace=args.include_trace,
    )
    out_path = _write_markdown_report(rendered, args.out_md)
    print(
        f"Saved markdown report to: {out_path}"
    )
