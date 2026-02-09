from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from services.agent.agent import run_agent


def _load_state(state_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not state_path:
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_path: str, state: Dict[str, Any]) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _print_top_results(state: Dict[str, Any]) -> None:
    tool_result = state.get("last_tool_result")
    if not isinstance(tool_result, dict):
        return
    results = tool_result.get("results")
    if not isinstance(results, list) or not results:
        return

    print("\nTop results:")
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item.get("opportunity_title") or "Untitled"
        score = item.get("score") or item.get("llm_score") or item.get("domain_score")
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        print(f"{idx}. {title} (score: {score_txt})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grant-matching agent")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument(
        "--state-json",
        default=None,
        help="Optional path to JSON state file to continue a prior interaction",
    )
    args = parser.parse_args()

    state = _load_state(args.state_json)
    result = run_agent(args.prompt, state=state)
    result_state = result.get("state") or {}

    state_path = args.state_json or ".agent_state.json"
    _save_state(state_path, result_state)
    print(f"State saved to: {state_path}")

    if result.get("type") == "clarification":
        print(result.get("question") or "")
        print(f"Hint: rerun with --state-json {state_path} to continue.")
        return

    if result.get("type") == "final":
        print(result.get("answer") or "")
        _print_top_results(result_state)
        return

    print("Unexpected response type.")


if __name__ == "__main__":
    main()
