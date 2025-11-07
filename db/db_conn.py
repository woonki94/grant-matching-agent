from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Example: postgresql+psycopg://user:pass@host:port/dbname
DATABASE_URL = "postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}".format(
    user=os.getenv("PGUSER", "ajoshi"),
    pwd=os.getenv("PGPASSWORD", "123456"),
    host=os.getenv("PGHOST", "127.0.0.1"),
    port=os.getenv("PGPORT", "5433"),
    db=os.getenv("PGDATABASE", "grants"),
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)