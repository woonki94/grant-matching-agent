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

# ───────────────────────────────────────────────
# Initialize DB
# ───────────────────────────────────────────────
def init_database() -> None:
    print("Creating database tables (if not exist)...")
    Base.metadata.create_all(engine)

    print("All tables ready.")

if __name__ == "__main__":
    init_database()