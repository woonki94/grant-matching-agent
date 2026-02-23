from __future__ import annotations

import argparse
from textwrap import fill

from dto.llm_response_dto import FacultyRecsOut
from services.justification.single_justification_generator import SingleJustificationGenerator


def print_faculty_recs(out: FacultyRecsOut, email: str, *, width: int = 92, show_full_id: bool = True) -> None:
    print("\n" + "=" * width)
    print(f"Faculty: {out.faculty_name}  <{email}>")
    print(f"Top {len(out.recommendations)} opportunities:")
    print("-" * width)

    for i, rec in enumerate(out.recommendations, start=1):
        llm_score = float(rec.llm_score)
        if llm_score < 0.30:
            label = "sucks"
        elif llm_score < 0.50:
            label = "bad"
        elif llm_score < 0.70:
            label = "good"
        elif llm_score < 0.85:
            label = "great"
        else:
            label = "fantastic"

        oid = rec.opportunity_id
        oid_disp = oid if show_full_id else (oid[:8] + "…" + oid[-6:])
        score_line = f"domain={rec.domain_score:.2f} | llm={rec.llm_score:.2f} | label={label}"

        print(f"{i:>2}. {rec.title}")
        print(f"    ID: {oid_disp}")
        if getattr(rec, "agency", None):
            print(f"    Agency: {rec.agency}")
        print(f"    Scores: {score_line}")
        print("    Why it matches:")
        for b in rec.why_good_match:
            print("      • " + fill((b or "").strip(), width=width, subsequent_indent="        ").lstrip())
        print("    Suggested pitch:")
        print("      " + fill((rec.suggested_pitch or "").strip(), width=width, subsequent_indent="        ").lstrip())
        print("-" * width)


def main(email: str, k: int) -> None:
    out = SingleJustificationGenerator().generate_faculty_recs(email=email, k=k)
    print_faculty_recs(out, email, show_full_id=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate top opportunity recommendations for a faculty by email")
    parser.add_argument("--email", required=True, help="Faculty email address (must exist in DB)")
    parser.add_argument("--k", type=int, default=5, help="Top-K opportunities to explain (default=5)")
    args = parser.parse_args()
    main(email=args.email.strip(), k=args.k)
