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
import db.models

from sqlalchemy import text

# ───────────────────────────────────────────────
# Clear All Tables
# ───────────────────────────────────────────────
def clear_all_data() -> None:
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            print("Clearing all data from tables...")
            # Disable foreign key constraints
            conn.execute(text("SET session_replication_role = 'replica';"))
            for table in reversed(Base.metadata.sorted_tables):
                print(f"Deleting from {table.name}...")
                conn.execute(table.delete())
            # Re-enable constraints
            conn.execute(text("SET session_replication_role = 'origin';"))
            trans.commit()
            print("All data cleared successfully.")
        except Exception as e:
            trans.rollback()
            print("Error clearing data:", e)

if __name__ == "__main__":
    clear_all_data()