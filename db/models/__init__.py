# db/models/__init__.py
from .opportunity import Opportunity

from .faculty import Faculty

from .match_result import MatchResult

__all__ = [
    "Opportunity",
    "Faculty",
    "MatchResult",
]