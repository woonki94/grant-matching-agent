from __future__ import annotations

from typing import Any, Callable, Dict, List


Tool = Dict[str, Any]


def _find_relevant_grants(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    from services.search.search_grants import search_grants

    return search_grants(**tool_input)


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
    return fn(tool_input)
