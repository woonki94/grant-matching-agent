from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from logging_setup import setup_logging
from services.justification.single_justification_generator import SingleJustificationGenerator


def main(email: str, k: int) -> None:
    out = SingleJustificationGenerator().run(email=email, k=k)
    if hasattr(out, "model_dump"):
        payload = out.model_dump()
    elif isinstance(out, dict):
        payload = out
    else:
        payload = {"raw_output": str(out)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    setup_logging("justification")
    parser = argparse.ArgumentParser(description="Generate raw LLM recommendation output for a faculty by email")
    parser.add_argument("--email", required=True, help="Faculty email address (must exist in DB)")
    parser.add_argument("--k", type=int, default=5, help="Top-K opportunities to send to the LLM (default=5)")
    args = parser.parse_args()
    main(email=args.email.strip(), k=args.k)
