# db/models/__init__.py
from .opportunity import Opportunity

from .faculty import Faculty

from .match_result import MatchResult

from .user import User

__all__ = [
    "Opportunity",
    "Faculty",
    "MatchResult",
    "User",
]