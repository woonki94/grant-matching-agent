from __future__ import annotations

import argparse
from pathlib import Path

from services.justification.group_justification_generator import GroupJustificationGenerator
from utils.report_renderer import write_markdown_report, render_markdown_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group justification generation")
    parser.add_argument(
        "--email",
        action="append",
        required=True,
        help="Faculty email. Repeat --email for multiple entries.",
    )
    parser.add_argument("--team-size", type=int, default=3, help="Team size (default: 3)")
    parser.add_argument("--limit-rows", type=int, default=200, help="Max number of match rows to scan")
    parser.add_argument("--opp-id", action="append", help="Target opportunity id. Repeatable.")
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Output markdown file path (default: outputs/justification_reports/auto-generated)",
    )
    parser.add_argument("--include-trace", action="store_true", help="Include trace output in result payload")
    args = parser.parse_args()

    group_justification = GroupJustificationGenerator()
    result = group_justification.run_justifications_from_group_results(
        faculty_emails=args.email,
        team_size=args.team_size,
        opp_ids=args.opp_id,
        limit_rows=args.limit_rows,
        include_trace=args.include_trace,
    )

    rendered = render_markdown_report(result)
    out_path = write_markdown_report(PROJECT_ROOT, rendered, args.out_md)
    print(f"Saved markdown report to: {out_path}")


if __name__ == "__main__":
    main()
