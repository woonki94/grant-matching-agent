import os
import sys

# ───────────────────────────────────────────────
# Make sure the project root is on sys.path
# ───────────────────────────────────────────────
if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from data.models.models_grant import Base           # Main Base used by all models
    import data.models.models_keyword                # <-- import your keyword model so it registers!
    from data.db_conn import engine
else:
    from data.models.models_grant import Base
    import data.models.models_keyword
    from .db_conn import engine


# ───────────────────────────────────────────────
# Initialize DB
# ───────────────────────────────────────────────
def init_database() -> None:
    print("Creating database tables (if not exist)...")
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_database()