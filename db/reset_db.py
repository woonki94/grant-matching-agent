import os
import sys
from sqlalchemy import text

from init_db import init_database

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

# ───────────────────────────────────────────────
# Drop All Tables
# ───────────────────────────────────────────────
def drop_database(confirm: bool = True) -> None:
    """Drop all tables from the database."""
    if confirm:
        ans = input(
            "This will DROP ALL TABLES in the current database. Continue? (y/N): "
        ).strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    print("Dropping all tables...")

    with engine.connect() as conn:
        # Disable foreign key checks (Postgres-safe)
        conn.execute(text("SET session_replication_role = 'replica';"))

        # Drop all tables
        Base.metadata.drop_all(bind=engine)

        # Re-enable constraints
        conn.execute(text("SET session_replication_role = 'origin';"))
        print("All tables dropped successfully.")


if __name__ == "__main__":
    drop_database()
    init_database()
