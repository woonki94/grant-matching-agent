import os
import sys

# ───────────────────────────────────────────────
# Ensure project root on sys.path
# ───────────────────────────────────────────────
if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

# Shared Base + engine
from db.base import Base
from db.db_conn import engine

# Import ALL model modules so their tables register on Base
from db.models.keywords_grant import Keyword
from db.models.keywords_faculty import FacultyKeyword

from db.models.grant import Opportunity
from db.models.faculty import Faculty

# ───────────────────────────────────────────────
# Initialize DB
# ───────────────────────────────────────────────
def init_database() -> None:
    print("Creating database tables (if not exist)...")
    print("Models loaded tables:", list(Base.metadata.tables.keys()))

    Base.metadata.create_all(engine)

    print("All tables ready.")

#TODO: Automate Database creation
if __name__ == "__main__":
    print("Known models:", sorted(k for k in Base.registry._class_registry.keys()
                                  if isinstance(k, str)))
    init_database()