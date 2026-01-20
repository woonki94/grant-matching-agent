from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

#TODO: to config
DATABASE_URL = "postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}".format(
    user=os.getenv("PGUSER", "ec2-user"),
    pwd=os.getenv("PGPASSWORD", "awsaws"),
    host=os.getenv("PGHOST", "127.0.0.1"),
    port=os.getenv("PGPORT", "5432"),
    db=os.getenv("PGDATABASE", "grants_v2"),
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
