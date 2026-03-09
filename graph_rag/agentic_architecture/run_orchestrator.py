from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on sys.path for direct script execution.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from graph_rag.agentic_architecture.orchestrator import GraphRagOrchestrator
from graph_rag.agentic_architecture.state import AgenticRequest
from graph_rag.common import json_ready, load_dotenv_if_present


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simple agentic architecture scenario.")
    parser.add_argument("--scenario", type=str, default="faculty_one_to_one_match")
    parser.add_argument("--email", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--include-closed", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    load_dotenv_if_present()

    orchestrator = GraphRagOrchestrator()
    request = AgenticRequest(
        scenario=str(args.scenario or "").strip(),
        email=str(args.email or "").strip().lower(),
        threshold=(None if args.threshold is None else float(args.threshold)),
        top_k=max(0, int(args.top_k or 0)),
        include_closed=bool(args.include_closed),
    )
    out = orchestrator.run(request)
    print(json.dumps(json_ready(out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
