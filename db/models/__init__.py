# db/models/__init__.py
from .opportunity import Opportunity, Attachment
from .keywords_opportunity import Keyword

from .faculty import Faculty
from .keywords_faculty import FacultyKeyword

from .match_result import MatchResult

__all__ = [
    "Opportunity",
    "Attachment",
    "Keyword",
    "Faculty",
    "FacultyKeyword",
    "MatchResult",
]