from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolInputError(Exception):
    tool_name: str
    message: str
    missing_fields: Optional[List[str]] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "message": self.message,
            "missing_fields": self.missing_fields,
            "details": self.details,
        }


MISSING_FIELD_FRIENDLY_BY_TOOL = {
    "find_additional_collaborators": {
        "need_y": "how many additional collaborators you need to add",
        "team_size": "the final team size",
        "opp_ids": "the opportunity ID(s)",
        "faculty_emails": "at least one email for the current team members",
    },
    "find_team_for_grant": {
        "opp_ids": "the opportunity ID(s)",
        "team_size": "the final team size",
    },
}
