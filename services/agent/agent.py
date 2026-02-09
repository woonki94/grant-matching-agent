from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from config import get_llm_client
from services.agent.tools import call_tool, list_tools
from services.prompts.agent_planner_prompt import AGENT_PLANNER_PROMPT


logger = logging.getLogger(__name__)


def _ensure_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(state or {})
    out.setdefault("messages", [])
    out.setdefault("slots", {})
    return out


def _merge_state_updates(state: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(state.get(k), dict):
            state[k].update(v)
        else:
            state[k] = v


def _parse_json(text: str) -> Dict[str, Any]:
    return json.loads(text)


def _extract_json(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return ""


def _tools_for_prompt(tools: list[dict]) -> list[dict]:
    clean = []
    for t in tools:
        item = dict(t)
        item.pop("fn", None)
        clean.append(item)
    return clean


def run_agent(user_prompt: str, state: Optional[dict] = None, max_steps: int = 5) -> dict:
    state = _ensure_state(state)
    state["messages"].append({"role": "user", "content": user_prompt})

    llm = get_llm_client().build()
    chain = AGENT_PLANNER_PROMPT | llm

    available_tools = _tools_for_prompt(list_tools())

    for step in range(max_steps):
        logger.info("agent_step=%d", step + 1)

        resp = chain.invoke(
            {
                "user_prompt": user_prompt,
                "conversation_state": json.dumps(state, ensure_ascii=False),
                "available_tools": json.dumps(available_tools, ensure_ascii=False),
            }
        )
        content = getattr(resp, "content", resp)
        logger.debug("planner_raw=%s", content)

        try:
            json_text = _extract_json(content)
            plan = _parse_json(json_text)
        except Exception as exc:
            logger.exception("planner_parse_error: %s", exc)
            return {
                "type": "clarification",
                "question": "I could not parse the plan. Could you rephrase your request?",
                "state": state,
            }

        state_updates = plan.get("state_updates") or {}
        _merge_state_updates(state, state_updates)

        action = plan.get("action")
        if action == "ask_user":
            question = plan.get("question") or "Could you clarify?"
            return {"type": "clarification", "question": question, "state": state}

        if action == "call_tool":
            tool_name = plan.get("tool_name")
            tool_input = plan.get("tool_input") or {}
            if not tool_name:
                return {
                    "type": "clarification",
                    "question": "Which tool should I use?",
                    "state": state,
                }

            logger.info("calling_tool=%s", tool_name)
            try:
                result = call_tool(tool_name, tool_input)
            except Exception as exc:
                logger.exception("tool_error: %s", exc)
                return {
                    "type": "clarification",
                    "question": "I hit an error running the tool. Can you try again?",
                    "state": state,
                }

            state["last_tool_result"] = result
            continue

        if action == "finish":
            answer = plan.get("final_answer") or ""
            return {"type": "final", "answer": answer, "state": state}

        logger.warning("unknown_action=%s", action)

    # Max steps fallback
    slots = state.get("slots") or {}
    missing = []
    if not slots.get("query_text"):
        missing.append("research topic or query text")
    if not slots.get("top_k"):
        missing.append("how many results you want (top_k)")

    if missing:
        question = "I need: " + "; ".join(missing) + "."
    else:
        question = "What should I do next with the current context?"

    return {
        "type": "clarification",
        "question": question,
        "state": state,
    }
