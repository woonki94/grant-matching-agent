from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time


def render_markdown_report(results: List[Dict[str, Any]]) -> str:
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
        lines.append("")
        lines.append("---")
        lines.append("")

        if r.get("error"):
            lines.append("## ⚠️ Processing Error")
            lines.append("")
            lines.append(r["error"])
            lines.append("")
            lines.append("## Recommended Action")
            lines.append("Review this opportunity manually or rerun justification generation.")
            lines.append("")
            continue

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

        lines.append("## What This Grant Is About")
        lines.append("")
        grant_quick_explanation = str(just.get("one_paragraph") or "").strip()
        if grant_quick_explanation:
            lines.append(grant_quick_explanation)
        else:
            lines.append("No quick grant explanation was generated.")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## Faculty Roles")
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

        for m in r.get("team_members", []):
            fid = m.get("faculty_id")
            name = m.get("faculty_name") or f"Faculty {fid}"
            bullets = strengths_by_faculty.get(int(fid)) if isinstance(fid, int) else None
            lines.append(f"### What {name} Can Do for This Grant")
            lines.append("")
            if bullets:
                for b in bullets[:10]:
                    lines.append(f"- {_format_strength_bullet(b)}")
            lines.append("")

        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## Why This Might Not Work")
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
        lines.append("## Recommended Action")
        lines.append("")
        recommendation = (just.get("recommendation") or "").strip()
        if recommendation:
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


def write_markdown_report(project_root: Path, markdown_text: str, output_path: Optional[str] = None) -> Path:
    if output_path:
        out = Path(output_path).expanduser()
    else:
        out = project_root / "outputs" / "justification_reports" / f"group_justification_{int(time.time())}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown_text, encoding="utf-8")
    return out
