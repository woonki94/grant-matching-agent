from __future__ import annotations

from typing import Any, Callable, Dict, List


Tool = Dict[str, Any]


from services.agent.tool_errors import ToolInputError


def _find_relevant_grants(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    from services.search.search_grants import search_grants

    return search_grants(**tool_input)


def _find_additional_collaborators(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    from services.agent.tool_impl.find_additional_collaborators import (
        find_additional_collaborators,
    )

    faculty_emails = tool_input.get("faculty_emails") or []
    opp_ids = tool_input.get("opp_ids")
    need_y = tool_input.get("need_y")
    team_size = tool_input.get("team_size")

    return find_additional_collaborators(
        faculty_emails=faculty_emails,
        opp_ids=opp_ids,
        team_size=int(team_size),
    )


def _find_team_for_grant(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    from services.agent.tool_impl.find_team_for_grant import (
        find_team_for_grant,
    )

    opp_ids = tool_input.get("opp_ids") or []
    team_size = tool_input.get("team_size")
    return find_team_for_grant(
        opp_ids=opp_ids,
        team_size=int(team_size),
    )


_TOOLS: Dict[str, Tool] = {
    "find_relevant_grants": {
        "name": "find_relevant_grants",
        "description": "Given a research topic and optional URLs/filters, returns ranked grant opportunities",
        "input_schema": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string", "description": "Research topic or query text"},
                "top_k": {"type": "integer", "minimum": 1, "description": "Max number of results"},
                "user_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional URLs to include as extra context",
                },
                "agency": {"type": "string", "description": "Filter by agency"},
                "category": {"type": "string", "description": "Filter by category"},
                "status": {"type": "string", "description": "Filter by opportunity status"},
            },
            "required": ["query_text"],
            "additionalProperties": False,
        },
        "fn": _find_relevant_grants,
    }
    ,
    "find_additional_collaborators": {
        "name": "find_additional_collaborators",
        "description": "Given existing team faculty emails, a/or set of opportunity ID, and a desired final team size, recommend additional faculty collaborators.",
        "input_schema": {
            "type": "object",
            "properties": {
                "faculty_emails": {"type": "array", "items": {"type": "string"}},
                "opp_ids": {"type": "array", "items": {"type": "string"}},
                "need_y": {"type": "integer", "minimum": 1},
                "team_size": {"type": "integer", "minimum": 2},
            },
            "required": ["faculty_emails", "need_y"],
            "additionalProperties": False,
        },
        "fn": _find_additional_collaborators,
    }
    ,
    "find_team_for_grant": {
        "name": "find_team_for_grant",
        "description": "Given an opportunity ID or title and desired team size, recommend a full faculty team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "opp_ids": {"type": "array", "items": {"type": "string"}},
                "team_size": {"type": "integer", "minimum": 1},
            },
            "required": ["opp_ids", "team_size"],
            "additionalProperties": False,
        },
        "fn": _find_team_for_grant,
    }
}


def list_tools() -> List[Tool]:
    return list(_TOOLS.values())


def get_tool(name: str) -> Tool:
    tool = _TOOLS.get(name)
    if not tool:
        raise KeyError(f"Unknown tool: {name}")
    return tool


def call_tool(name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    tool = get_tool(name)
    fn: Callable[[Dict[str, Any]], Dict[str, Any]] = tool["fn"]
    try:
        if name == "find_additional_collaborators":
            faculty_emails = tool_input.get("faculty_emails")
            if not faculty_emails:
                raise ToolInputError(
                    tool_name=name,
                    message="Please provide at least one faculty email for the current team.",
                    missing_fields=["faculty_emails"],
                )
            faculty_emails = [e for e in faculty_emails if str(e).strip()]
            if not faculty_emails:
                raise ToolInputError(
                    tool_name=name,
                    message="Please provide at least one faculty email for the current team.",
                    missing_fields=["faculty_emails"],
                )

            need_y = tool_input.get("need_y")
            try:
                need_y_int = int(need_y)
            except Exception:
                need_y_int = None
            if need_y_int is None or need_y_int < 1:
                raise ToolInputError(
                    tool_name=name,
                    message="How many additional collaborators do you need to add?",
                    missing_fields=["need_y"],
                )

            team_size = tool_input.get("team_size")
            if team_size is None:
                tool_input["team_size"] = len(faculty_emails) + need_y_int
            else:
                try:
                    team_size_int = int(team_size)
                except Exception:
                    team_size_int = None
                expected = len(faculty_emails) + need_y_int
                if team_size_int is None or team_size_int != expected:
                    raise ToolInputError(
                        tool_name=name,
                        message="Team size doesn't match current team + additional collaborators. Please confirm the final team size.",
                        details={
                            "current_team_size": len(faculty_emails),
                            "need_y": need_y_int,
                            "expected_team_size": expected,
                            "provided_team_size": team_size,
                        },
                    )

            # opp_ids optional; pass through as None if missing/empty
            if not tool_input.get("opp_ids"):
                tool_input["opp_ids"] = None
        if name == "find_team_for_grant":
            opp_ids = tool_input.get("opp_ids")
            if not opp_ids:
                raise ToolInputError(
                    tool_name=name,
                    message="Please provide the opportunity ID or title.",
                    missing_fields=["opp_ids"],
                )
            team_size = tool_input.get("team_size")
            try:
                team_size_int = int(team_size)
            except Exception:
                team_size_int = None
            if team_size_int is None or team_size_int < 1:
                raise ToolInputError(
                    tool_name=name,
                    message="What final team size do you want?",
                    missing_fields=["team_size"],
                )

        return fn(tool_input)
    except ToolInputError as exc:
        return {"error": exc.to_dict()}
